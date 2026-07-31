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

## Supplementary dataset: D2D dashed markers
- Roboflow workspace: `d2d`
- Project: `lane-detection-tqgmk`
- Version: 2
- License: **CC BY 4.0** (not MIT like the primary dataset — requires
  attribution if the trained model is ever published/distributed)
- URL: https://universe.roboflow.com/d2d/lane-detection-tqgmk/dataset/2
- 4 classes: `crossing`, `dashed-lane`, `full-lane`, `transverse-lane`

The original `lane-detection-awvt7` set is sourced from Korean roads with
only solid lane markings, which is why the trained model was failing to
detect dashed lane lines. This dataset adds real dashed-marker examples.

Added via two scripts, gitignored the same way as `Lane-Detection-2/`:
- `download_d2d_dataset.py` — downloads the latest version to
  `D2D-Lane-Detection/`.
- `remap_d2d_labels.py` — D2D's 4 classes don't match this project's single
  `lane` class. `dashed-lane`/`full-lane` are real lane-boundary examples and
  get remapped to class 0; `crossing`/`transverse-lane` are crosswalk/
  stop-line style boxes with a different shape, so those annotation lines are
  dropped entirely rather than mislabeled as `lane`. Idempotent via a
  `.remapped` marker file in `D2D-Lane-Detection/`.

`train.py` now points at `combined_data.yaml` (repo root, tracked in git)
instead of `Lane-Detection-2/data.yaml` directly — it lists both dataset
directories' image folders as a combined `train`/`val` set. It uses paths
relative to the repo root (unlike the Roboflow-generated `data.yaml`, which
hardcodes an absolute path for whatever machine downloaded it), so no
per-machine edits are needed there — just make sure both dataset folders
have been downloaded before training.

Verified locally (Mac) that `ultralytics.data.utils.check_det_dataset`
resolves `combined_data.yaml` into a single 1-class (`lane`) dataset combining
both sources before this was committed.

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
- [ ] Run `python download_d2d_dataset.py && python remap_d2d_labels.py` to
      fetch and remap the supplementary dashed-marker dataset — `train.py`
      now trains on both datasets via `combined_data.yaml`.
