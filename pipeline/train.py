"""Fine-tune flan-t5-base to rewrite professional cancer text in plain words.

Why a plain PyTorch loop instead of the Hugging Face Trainer: every step
(batch -> loss -> backward -> update) is visible in about 30 lines, so a team
member can explain it to a judge, and it does not depend on library APIs that
change between versions.

Run (see docs/TRAINING.md):  python -m pipeline.train
The model only ever sees data/pdq_pairs.jsonl rows whose split is "train" or
"val". The "test" split is never loaded here, so evaluate.py stays honest.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import subprocess
import sys
import time
from pathlib import Path

from backend import config

TRAIN_SPLIT, VAL_SPLIT = "train", "val"  # "test" is deliberately absent


def load_pairs(path: "Path | list[Path]", min_similarity: float) -> tuple[list[dict], list[dict]]:
    """Return (train, val) rows with similarity >= min_similarity.

    Why filter: a pair that is only loosely aligned teaches the model to say
    things the source never said, which is exactly what our faithfulness rule
    forbids. Rows from any other split (test) are skipped on purpose.
    """
    train, val = [], []
    for one in (path if isinstance(path, (list, tuple)) else [path]):  # several files = one combined pool
        with open(one, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row["similarity"] < min_similarity:
                    continue
                if row["split"] == TRAIN_SPLIT:
                    train.append(row)
                elif row["split"] == VAL_SPLIT:
                    val.append(row)
    return train, val


def with_prefix(source: str) -> str:
    """Prepend the task prefix; T5 models use it to know which job to do."""
    return config.TASK_PREFIX + source


def make_collate(tokenizer):
    """Build a function that turns a list of rows into padded tensors."""
    def collate(rows: list[dict]) -> dict:
        enc = tokenizer([with_prefix(r["source"]) for r in rows], max_length=config.MAX_INPUT_TOKENS,
                        truncation=True, padding=True, return_tensors="pt")
        lab = tokenizer(text_target=[r["target"] for r in rows], max_length=config.MAX_OUTPUT_TOKENS,
                        truncation=True, padding=True, return_tensors="pt")["input_ids"]
        lab[lab == tokenizer.pad_token_id] = -100  # -100 = "ignore this position in the loss"
        enc["labels"] = lab
        return dict(enc)
    return collate


def git_commit() -> str:
    """Current git commit hash (so a checkpoint can be traced to its code)."""
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=config.ROOT_DIR,
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def pick_device(force_cpu: bool):
    """Choose device and precision; print what we found.

    bf16 (not fp16) because T5 activations overflow in fp16 and give NaN loss.
    With two GPUs we take the one with the most memory.
    """
    import torch
    if force_cpu or not torch.cuda.is_available():
        print("Device: CPU (precision fp32)" + ("" if force_cpu else "  -- no CUDA GPU visible to torch!"))
        return torch.device("cpu"), False
    idx = max(range(torch.cuda.device_count()), key=lambda i: torch.cuda.get_device_properties(i).total_memory)
    props = torch.cuda.get_device_properties(idx)
    use_bf16 = torch.cuda.is_bf16_supported()
    print(f"Device: cuda:{idx} {props.name}, {props.total_memory / 2**30:.1f} GiB VRAM, "
          f"precision {'bf16' if use_bf16 else 'fp32'}")
    return torch.device(f"cuda:{idx}"), use_bf16


def evaluate_loss(model, loader, device, use_bf16: bool) -> float:
    """Average loss on the validation pairs (lower = better, no weight updates)."""
    import torch
    model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_bf16):
                loss = model(**batch).loss
            total += loss.item() * batch["labels"].shape[0]
            n += batch["labels"].shape[0]
    return total / max(n, 1)


def save_best(model, tokenizer, out: Path, meta: dict) -> None:
    """Write model, tokenizer and metadata so the folder is self-contained."""
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    tokenizer.save_pretrained(out)
    (out / "train_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def write_log(out: Path, history: list[dict]) -> None:
    """Save per-epoch losses as JSON and CSV (for the loss-curve plot in EVAL.md)."""
    out.mkdir(parents=True, exist_ok=True)
    (out / "train_log.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    with open(out / "loss_curve.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["epoch", "step", "train_loss", "val_loss"])
        w.writeheader()
        w.writerows(history)


def fmt_eta(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m}m{s:02d}s"


def train(args: argparse.Namespace) -> Path:
    """Run the whole fine-tune and return the checkpoint folder."""
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer, T5ForConditionalGeneration, get_linear_schedule_with_warmup

    random.seed(config.RANDOM_SEED)
    torch.manual_seed(config.RANDOM_SEED)
    train_rows, val_rows = load_pairs(args.pairs, args.min_similarity)
    print(f"Pairs at similarity >= {args.min_similarity}: train={len(train_rows)} val={len(val_rows)}")
    if not train_rows or not val_rows:
        sys.exit("No train/val pairs left after filtering; lower --min-similarity.")

    device, use_bf16 = pick_device(args.cpu)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = T5ForConditionalGeneration.from_pretrained(args.model).to(device)
    collate = make_collate(tokenizer)
    gen = torch.Generator().manual_seed(config.RANDOM_SEED)
    train_dl = DataLoader(train_rows, batch_size=args.batch_size, shuffle=True, collate_fn=collate, generator=gen)
    val_dl = DataLoader(val_rows, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    accum = config.TRAIN_GRAD_ACCUM
    steps_per_epoch = -(-len(train_dl) // accum)
    total_steps = min(args.epochs * steps_per_epoch, args.max_steps or 10**9)
    no_decay = ("bias", "layer_norm", "LayerNorm")
    groups = [
        {"params": [p for n, p in model.named_parameters() if not any(k in n for k in no_decay)],
         "weight_decay": config.TRAIN_WEIGHT_DECAY},
        {"params": [p for n, p in model.named_parameters() if any(k in n for k in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(groups, lr=args.lr)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(config.TRAIN_WARMUP_RATIO * total_steps), total_steps)
    print(f"Optimizer steps: {total_steps} ({steps_per_epoch}/epoch, batch {args.batch_size} x accum {accum})")

    meta = {"base_model": args.model, "min_similarity": args.min_similarity,
            "git_commit": git_commit(), "train_pairs": len(train_rows), "val_pairs": len(val_rows),
            "learning_rate": args.lr, "pairs_path": [str(p) for p in args.pairs], "batch_size": args.batch_size,
            "grad_accum": accum, "epochs_max": args.epochs, "seed": config.RANDOM_SEED,
            "task_prefix": config.TASK_PREFIX, "bf16": use_bf16, "max_steps": args.max_steps}
    out: Path = args.out
    history: list[dict] = []
    best, bad_epochs, step, t0 = float("inf"), 0, 0, time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        run_loss, run_n = 0.0, 0
        optimizer.zero_grad()
        for i, batch in enumerate(train_dl, 1):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_bf16):
                loss = model(**batch).loss
            (loss / accum).backward()
            run_loss += loss.item()
            run_n += 1
            if i % accum == 0 or i == len(train_dl):
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN_MAX_GRAD_NORM)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                step += 1
                if step % 10 == 0 or step == total_steps:
                    rate = step / (time.time() - t0)
                    print(f"epoch {epoch} step {step}/{total_steps} loss {run_loss / run_n:.4f} "
                          f"{rate:.2f} steps/s ETA {fmt_eta((total_steps - step) / rate)}", flush=True)
                if step >= total_steps:
                    break
        val_loss = evaluate_loss(model, val_dl, device, use_bf16)
        train_loss = run_loss / max(run_n, 1)
        history.append({"epoch": epoch, "step": step, "train_loss": round(train_loss, 5), "val_loss": round(val_loss, 5)})
        print(f"== epoch {epoch}: train {train_loss:.4f} val {val_loss:.4f}", flush=True)
        if val_loss < best:
            best, bad_epochs = val_loss, 0
            save_best(model, tokenizer, out, {**meta, "best_epoch": epoch, "best_val_loss": round(val_loss, 5)})
            print(f"   new best -> saved to {out}")
        else:
            bad_epochs += 1
            if bad_epochs >= config.TRAIN_EARLY_STOP_PATIENCE:
                print("Early stopping: val loss stopped improving.")
                break
        write_log(out, history)
        if step >= total_steps:
            break
    write_log(out, history)
    print(f"Done in {fmt_eta(time.time() - t0)}. Best val loss {best:.4f}. Checkpoint: {out}")
    return out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune flan-t5-base on PDQ pairs.")
    p.add_argument("--min-similarity", type=float, default=config.TRAIN_MIN_SIMILARITY)
    p.add_argument("--lr", type=float, default=config.TRAIN_LEARNING_RATE)
    p.add_argument("--pairs", type=Path, nargs="+", default=[config.PDQ_PAIRS_PATH],
                   help="one or more pairs JSONL files, pooled (default: paragraph pairs)")
    p.add_argument("--model", default=config.BASE_MODEL_NAME, help="T5-family base model (recorded in metadata)")
    p.add_argument("--epochs", type=int, default=config.TRAIN_EPOCHS)
    p.add_argument("--batch-size", type=int, default=config.TRAIN_BATCH_SIZE)
    p.add_argument("--out", type=Path, default=config.CHECKPOINT_DIR / config.TRAIN_CHECKPOINT_NAME)
    p.add_argument("--max-steps", type=int, default=None, help="stop after N weight updates (smoke test)")
    p.add_argument("--cpu", action="store_true", help="force CPU even if a GPU exists")
    return p.parse_args(argv)


if __name__ == "__main__":
    train(parse_args())
