"""Pure helpers of pipeline/train_lora.py and the multi-file loader (no model, no GPU)."""
import json

from backend import config
from pipeline import baselines, train, train_lora


def test_chat_messages_carry_instruction_and_source():
    m = train_lora.chat_messages("Stage III.")
    assert m[0]["role"] == "user" and m[0]["content"].startswith(config.LORA_INSTRUCTION) and m[0]["content"].endswith("Stage III.")


def test_j2_recipe_matches_adapted_settings_and_never_uses_test():
    r = train_lora.RECIPES["j2"]
    assert (r["r"], r["alpha"], r["lr"]) == (64, 16, 2e-4) and len(r["sweep"]) <= 2


def test_load_pairs_pools_files_and_skips_test(tmp_path):
    rows = [{"similarity": 0.9, "split": s, "source": "a", "target": "b"} for s in ("train", "val", "test")]
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    for p in (a, b):
        p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    tr, va = train.load_pairs([a, b], 0.7)
    assert (len(tr), len(va)) == (2, 2)


def test_is_causal(tmp_path):
    assert not baselines.is_causal(tmp_path)
    (tmp_path / "train_metadata.json").write_text(json.dumps({"model_type": "causal"}), encoding="utf-8")
    assert baselines.is_causal(tmp_path)
