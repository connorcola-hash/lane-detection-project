"""
Extract candidate training frames from raw VIOFO dashcam footage.

Frames are only saved when they differ meaningfully from the last *saved*
frame. That comparison state is global across the whole chronological run of
clips (never reset per-file), so a parked stretch spanning several
loop-recorded clips is treated as one continuous static period instead of
producing fresh "distinct" frames at every file boundary.
"""

import time
from pathlib import Path

import cv2
import numpy as np

SOURCE_DIR = Path("/Volumes/VIOFO/DCIM/Movie")
OUTPUT_DIR = Path("output/frames")

STRIDE = 10             # analyze every Nth decoded frame (30fps source -> ~3 checks/sec)
DIFF_THRESHOLD = 12.0   # mean abs pixel diff (0-255) on a downscaled grayscale frame
MIN_GAP_FRAMES = 30     # don't save more than once per this many source frames (~1s)
COMPARE_SIZE = (320, 180)


def list_clips():
    clips = sorted(
        p for p in SOURCE_DIR.glob("*.MP4") if not p.name.startswith("._")
    )
    return clips


def small_gray(frame):
    small = cv2.resize(frame, COMPARE_SIZE)
    return cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)


def process_clip(path, reference, frames_since_save, saved_count):
    cap = cv2.VideoCapture(str(path))
    frame_idx = 0
    clip_saved = 0

    while True:
        grabbed = cap.grab()
        if not grabbed:
            break

        if frame_idx % STRIDE == 0:
            ok, frame = cap.retrieve()
            if ok:
                gray = small_gray(frame)
                frames_since_save += STRIDE

                is_new = reference is None
                diff_score = None if is_new else float(np.abs(gray.astype(np.int16) - reference.astype(np.int16)).mean())

                if (is_new or diff_score > DIFF_THRESHOLD) and frames_since_save >= MIN_GAP_FRAMES:
                    out_name = f"{path.stem}_{frame_idx:06d}.jpg"
                    cv2.imwrite(str(OUTPUT_DIR / out_name), frame)
                    reference = gray
                    frames_since_save = 0
                    saved_count += 1
                    clip_saved += 1

        frame_idx += 1

    cap.release()
    return reference, frames_since_save, saved_count, clip_saved


def existing_saved_frames():
    return sorted(OUTPUT_DIR.glob("*.jpg"))


def clip_stem_of(frame_path):
    # frame filenames are "{TIMESTAMP}_{SEQNUM}_{FRAMEIDX}.jpg" -> clip stem is the first two parts
    parts = frame_path.stem.split("_")
    return "_".join(parts[:2])


def resume_state(clips):
    """Figure out where a prior run left off, using files already on disk
    (not any in-memory/log state, since that's lost on a crash)."""
    existing = existing_saved_frames()
    if not existing:
        return 0, None

    last_frame = existing[-1]
    last_stem = clip_stem_of(last_frame)
    clip_names = [c.stem for c in clips]
    try:
        resume_index = clip_names.index(last_stem)
    except ValueError:
        return 0, None

    reference = small_gray(cv2.imread(str(last_frame)))
    return resume_index, reference


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    clips = list_clips()
    print(f"Found {len(clips)} clips in {SOURCE_DIR} (RO/ event clips excluded by directory scope)", flush=True)

    resume_index, resumed_reference = resume_state(clips)
    saved_count = len(existing_saved_frames())
    if resume_index:
        print(f"Resuming from clip {resume_index + 1}/{len(clips)} ({saved_count} frames already saved)", flush=True)

    reference = resumed_reference
    frames_since_save = MIN_GAP_FRAMES  # allow the next candidate frame to save
    start = time.monotonic()

    for i, path in enumerate(clips[resume_index:], resume_index + 1):
        try:
            reference, frames_since_save, saved_count, clip_saved = process_clip(
                path, reference, frames_since_save, saved_count
            )
        except Exception as e:
            print(f"[{i}/{len(clips)}] {path.name}: skipped due to error ({e})", flush=True)
            continue

        elapsed = time.monotonic() - start
        print(f"[{i}/{len(clips)}] {path.name}: +{clip_saved} frames (total {saved_count}, {elapsed:.0f}s elapsed)", flush=True)

    print(f"Done. Saved {saved_count} frames from {len(clips)} clips to {OUTPUT_DIR}/", flush=True)


if __name__ == "__main__":
    main()
