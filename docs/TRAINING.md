# Training the summarizer (human-run)

Fine-tunes `google/flan-t5-base` (Apache 2.0) on `data/pdq_pairs.jsonl`. Tested on this PC:
RTX 5090 (32 GB), driver 591.86, CUDA 13.1, Windows. A real run is only a few minutes.

## 1. One-time setup (PowerShell, repo root)
Uses a separate venv so the CPU-only `.venv` used by the app/tests is untouched.
The `cu130` wheel matches the driver's CUDA 13.1 (cu132 would be too new; cu128 also works for Blackwell but has no torch 2.14).

```powershell
cd N:\Congressional-App-Challenge\study-search
py -3.11 -m venv .venv-train
.\.venv-train\Scripts\python -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu130
.\.venv-train\Scripts\python -m pip install numpy transformers==5.20.2 sentencepiece
.\.venv-train\Scripts\python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
Expect `2.14.1+cu130 True NVIDIA GeForce RTX 5090`. If it prints `False`, see the WSL section.
(The machine also has an RTX 4070 Ti; the script automatically picks the GPU with the most VRAM.)

## 2. Train
```powershell
.\.venv-train\Scripts\python -m pipeline.train
```
Defaults (all in `backend/config.py`, "Training" section): min similarity 0.60, batch 8 x accumulation 2, lr 3e-4, up to 5 epochs, early stop after 2 epochs without val-loss improvement, bf16.
Options: `--min-similarity 0.70`, `--epochs N`, `--batch-size N`, `--out PATH`, `--max-steps N` (quick test), `--cpu`.

Pairs used: min similarity 0.60 -> 1333 train / 161 val; 0.70 -> 1072 train / 125 val. The "test" split is never loaded.
Measured speed on the 5090: about 4.3 weight updates/s (16 pairs each). Full run (about 420 updates at 0.60) is roughly 2-4 minutes plus model download (~1 GB, first run only). The script prints an ETA every 10 steps.
First run downloads flan-t5-base into `%USERPROFILE%\.cache\huggingface`.

## 3. Output
`checkpoints/flan-t5-pdq/` (git-ignored): `model.safetensors`, tokenizer files, `train_metadata.json` (min_similarity, git commit, best epoch/val loss, settings), `train_log.json` and `loss_curve.csv` (epoch, step, train_loss, val_loss) for the loss plot in `docs/EVAL.md`.
The folder always holds the best-val-loss epoch so far, so stopping early is safe.

## 4. Stop / resume
Stop with Ctrl+C: the best checkpoint saved so far stays usable (only `train_log.json` may lag one epoch). There is no mid-run resume; runs are short, so just re-run (same seed). Use `--out checkpoints/flan-t5-pdq-0.70 --min-similarity 0.70` to keep several variants side by side. If CUDA runs out of memory, use `--batch-size 4`.

## WSL alternative (if Windows CUDA is not visible)
```bash
cd /mnt/n/Congressional-App-Challenge/study-search
python3.11 -m venv ~/venv-train && source ~/venv-train/bin/activate
pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu130
pip install numpy transformers==5.20.2 sentencepiece
python -m pipeline.train
```
WSL2 uses the Windows NVIDIA driver; do not install a Linux driver inside WSL. Check with `nvidia-smi` inside WSL first.
