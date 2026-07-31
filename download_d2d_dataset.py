"""
Download the D2D lane detection dataset (crossing / dashed-lane / full-lane /
transverse-lane) as a supplementary source of dashed-marker examples, which
the original Korean-roads dataset lacked.

Downloads whichever version is currently newest on Roboflow, since there's no
reason to pin to an older one for a dataset we don't otherwise depend on.
Run remap_d2d_labels.py afterward before training on it — its 4 classes need
filtering/remapping down to this project's single `lane` class first.
"""

import os

from roboflow import Roboflow

WORKSPACE = "d2d"
PROJECT_NAME = "lane-detection-tqgmk"
OUTPUT_DIR = "D2D-Lane-Detection"


def main():
    rf = Roboflow(api_key=os.environ["ROBOFLOW_API_KEY"])
    project = rf.workspace(WORKSPACE).project(PROJECT_NAME)

    versions = project.versions()
    latest = max(versions, key=lambda v: int(str(v.version).rsplit("/", 1)[-1]))
    version_number = int(str(latest.version).rsplit("/", 1)[-1])
    print(f"Downloading version {version_number} (latest of {len(versions)} available)")

    version = project.version(version_number)
    version.download("yolov8", location=OUTPUT_DIR)
    print(f"Done. Dataset in {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
