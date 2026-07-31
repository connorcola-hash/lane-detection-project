"""
Upload the selected + pre-labeled personal dashcam frames to a new Roboflow
project for model-assisted-labeling review. Predicted boxes are uploaded as
`is_prediction=True` so Roboflow treats them as drafts to confirm/correct,
not accepted ground truth.

Resumable: successfully uploaded frames are appended to UPLOAD_LOG, and a
rerun skips anything already logged.
"""

import os
from pathlib import Path

from roboflow import Roboflow
from roboflow.core.project import Project

WORKSPACE = "connors-workspace-khzty"
PROJECT_NAME = "personal-dashcam-lane-detection"
LABELMAP = {0: "lane"}

MANIFEST_PATH = Path("output/selected_frames.txt")
LABELS_DIR = Path("output/selected_labels")
UPLOAD_LOG = Path("output/uploaded.txt")
BATCH_NAME = "personal-footage-round1"


def already_uploaded():
    if not UPLOAD_LOG.exists():
        return set()
    return set(UPLOAD_LOG.read_text().splitlines())


def get_or_create_project(workspace, api_key):
    # workspace.project(name) hits a GET-single-project endpoint that 404s
    # with a permissions error for this project even though it demonstrably
    # exists (confirmed via workspace.projects()) -- a platform-side quirk,
    # not a timing issue. Work around it by constructing the Project object
    # directly from the listing data workspace already fetched, instead of
    # doing a second lookup call.
    full_id = f"{WORKSPACE}/{PROJECT_NAME}"
    match = next((p for p in workspace.project_list if p["id"] == full_id), None)
    if match:
        return Project(api_key, match, workspace.model_format)
    print(f"Creating new project '{PROJECT_NAME}'")
    return workspace.create_project(
        project_name=PROJECT_NAME,
        project_type="object-detection",
        project_license="MIT",
        annotation="lane",
    )


def main():
    api_key = os.environ["ROBOFLOW_API_KEY"]
    rf = Roboflow(api_key=api_key)
    workspace = rf.workspace(WORKSPACE)
    project = get_or_create_project(workspace, api_key)

    frames = [Path(p) for p in MANIFEST_PATH.read_text().splitlines() if p]
    done = already_uploaded()
    todo = [f for f in frames if f.name not in done]
    print(f"{len(frames)} frames total, {len(done)} already uploaded, {len(todo)} remaining")

    with open(UPLOAD_LOG, "a") as log:
        for i, frame_path in enumerate(todo, 1):
            label_path = LABELS_DIR / f"{frame_path.stem}.txt"
            has_boxes = label_path.exists() and label_path.stat().st_size > 0
            try:
                project.upload(
                    image_path=str(frame_path),
                    annotation_path=str(label_path) if has_boxes else None,
                    annotation_labelmap=LABELMAP,
                    is_prediction=True,
                    split="train",
                    batch_name=BATCH_NAME,
                )
            except Exception as e:
                print(f"[{i}/{len(todo)}] {frame_path.name}: FAILED ({e})", flush=True)
                continue

            log.write(frame_path.name + "\n")
            log.flush()

            if i % 50 == 0 or i == len(todo):
                print(f"[{i}/{len(todo)}] uploaded {frame_path.name}", flush=True)

    print("Done.")


if __name__ == "__main__":
    main()
