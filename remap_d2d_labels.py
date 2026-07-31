"""
Filter and remap the D2D dataset's 4 classes down to this project's single
`lane` class, in place, after download_d2d_dataset.py.

D2D labels `crossing` and `transverse-lane` as lane markings too, but those
are crosswalk/stop-line style boxes with a different shape than a lane
boundary — mixing them into the single `lane` class would teach the model
the wrong thing, so their annotation lines are dropped entirely (the image
itself is kept; it just loses those boxes). `dashed-lane` and `full-lane`
are true lane-boundary examples and get remapped to class 0.

Idempotent: writes a `.remapped` marker next to the dataset so reruns are a
no-op instead of double-filtering already-remapped labels.
"""

from pathlib import Path

import yaml

DATASET_DIR = Path("D2D-Lane-Detection")
KEEP_CLASSES = {"dashed-lane", "full-lane"}
MARKER = DATASET_DIR / ".remapped"


def remap_split(labels_dir, keep_indices):
    kept_boxes = 0
    dropped_boxes = 0
    touched_files = 0

    for label_path in labels_dir.glob("*.txt"):
        lines = label_path.read_text().splitlines()
        kept_lines = []
        for line in lines:
            if not line.strip():
                continue
            cls, *rest = line.split()
            if int(cls) in keep_indices:
                kept_lines.append(" ".join(["0", *rest]))
                kept_boxes += 1
            else:
                dropped_boxes += 1

        if len(kept_lines) != len(lines):
            touched_files += 1
        label_path.write_text("\n".join(kept_lines) + ("\n" if kept_lines else ""))

    return kept_boxes, dropped_boxes, touched_files


def main():
    if MARKER.exists():
        print(f"{MARKER} already exists, skipping (delete it to force a re-run)")
        return

    data_yaml_path = DATASET_DIR / "data.yaml"
    data = yaml.safe_load(data_yaml_path.read_text())
    names = data["names"]
    keep_indices = {i for i, name in enumerate(names) if name in KEEP_CLASSES}
    missing = KEEP_CLASSES - {names[i] for i in keep_indices}
    if missing:
        raise SystemExit(f"Expected classes not found in {data_yaml_path}: {missing} (found: {names})")

    for split in ("train", "valid", "test"):
        labels_dir = DATASET_DIR / split / "labels"
        if not labels_dir.is_dir():
            continue
        kept, dropped, touched = remap_split(labels_dir, keep_indices)
        print(f"{split}: kept {kept} boxes, dropped {dropped} boxes ({touched} files touched)")

    data["nc"] = 1
    data["names"] = ["lane"]
    data["_original_names"] = names
    data_yaml_path.write_text(yaml.safe_dump(data, sort_keys=False))

    MARKER.touch()
    print("Done. Wrote marker to prevent re-running on an already-remapped dataset.")


if __name__ == "__main__":
    main()
