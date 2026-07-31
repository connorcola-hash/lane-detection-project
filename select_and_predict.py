"""
Downsample the sampled dashcam frames to a labeling-sized subset and
pre-label them with the current best checkpoint, so Roboflow's review queue
starts from model-drafted boxes instead of blank frames.

Selection is a simple systematic stride over the full chronologically-sorted
frame list (frames are already deduplicated relative to each other by
sample_frames.py), which spreads the subset evenly across every clip/date/
lighting condition rather than clustering.
"""

from pathlib import Path

from ultralytics import YOLO

FRAMES_DIR = Path("output/frames")
LABELS_DIR = Path("output/selected_labels")
MANIFEST_PATH = Path("output/selected_frames.txt")
TARGET_COUNT = 1500


def select_frames():
    all_frames = sorted(FRAMES_DIR.glob("*.jpg"))
    stride = max(1, len(all_frames) // TARGET_COUNT)
    selected = all_frames[::stride]
    print(f"{len(all_frames)} candidate frames, stride {stride} -> {len(selected)} selected")
    return selected


def main():
    LABELS_DIR.mkdir(parents=True, exist_ok=True)
    selected = select_frames()
    MANIFEST_PATH.write_text("\n".join(str(p) for p in selected) + "\n")

    model = YOLO("checkpoints/best.pt")

    for i, frame_path in enumerate(selected, 1):
        label_path = LABELS_DIR / f"{frame_path.stem}.txt"
        if label_path.exists():
            continue

        result = model(str(frame_path), verbose=False)[0]
        lines = []
        for box in result.boxes:
            cls = int(box.cls.item())
            x, y, w, h = box.xywhn[0].tolist()
            lines.append(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
        label_path.write_text("\n".join(lines) + ("\n" if lines else ""))

        if i % 100 == 0 or i == len(selected):
            print(f"[{i}/{len(selected)}] predicted {frame_path.name} ({len(lines)} boxes)", flush=True)

    print(f"Done. {len(selected)} frames selected, labels in {LABELS_DIR}/, manifest at {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
