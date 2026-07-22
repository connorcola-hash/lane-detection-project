# Migration to Windows desktop (RTX 30-series)

## Dataset source
- Roboflow workspace: `detrlane`
- Project: `lane-detection-awvt7`
- Version: 2
- License: MIT
- URL: https://universe.roboflow.com/detrlane/lane-detection-awvt7/dataset/2
- 1 class: `lane`

## Plan
The dataset (`Lane-Detection-2/`, currently 2.7GB locally) is gitignored, not
committed. Re-download it directly on the Windows machine with
`download_dataset.py` instead of transferring it — faster than moving GBs
over a network/drive, and guarantees a fresh copy with paths that match the
new machine.

`download_dataset.py` now reads the Roboflow API key from the
`ROBOFLOW_API_KEY` environment variable instead of a hardcoded string — set
that env var on the Windows machine before running it.

## Current training state
- `imgsz`: 640
- `batch`: 2 (tuned down on the Mac to avoid OOM kills on 8GB unified memory —
  the RTX desktop has far more VRAM, so this should be raised, e.g. via
  ultralytics' `batch=-1` auto-batch, or a manual value once you know the
  card's VRAM)
- `device`: `'mps'` (Mac-specific — see Windows checklist below)
- Most recent checkpoint: `runs/detect/runs/train/yolov8n_43-3/weights/last.pt`
  — 3 of 43 configured epochs completed in that run (that run itself resumed
  from an earlier checkpoint already at epoch 7, so ~10 epochs of training
  lineage total). Copied into `checkpoints/last.pt` and `checkpoints/best.pt`
  so they sync through git.

## Windows setup checklist
- [ ] Install CUDA-enabled PyTorch (pick the build matching your installed
      CUDA version at https://pytorch.org/get-started/locally/) — the
      default pip install of `torch` may give you a CPU-only build.
- [ ] Set the `ROBOFLOW_API_KEY` environment variable.
- [ ] Run `python download_dataset.py` to fetch the dataset locally.
- [ ] Check `Lane-Detection-2/data.yaml` — Roboflow writes absolute paths for
      whatever machine downloads it, so confirm `train:`/`val:` point at the
      right location on the Windows filesystem.
- [ ] In `train.py`, change `device='mps'` to `device=0` (or `'cuda'`).
- [ ] Point the model load at `checkpoints/last.pt` instead of the old Mac
      absolute path, to resume from the synced checkpoint.
- [ ] Wrap the `model.train(...)` call in `if __name__ == '__main__':` —
      required on Windows because its multiprocessing start method (`spawn`)
      re-imports the script in each DataLoader worker; without the guard,
      training recurses/crashes on Windows in a way it doesn't on
      macOS/Linux (which use `fork`).
- [ ] Disable sleep during long training runs — Windows equivalent of
      `caffeinate`: `powercfg /change standby-timeout-ac 0` (or Settings >
      Power & sleep), and revert afterward if desired.
