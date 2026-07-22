import os
from roboflow import Roboflow
rf = Roboflow(api_key=os.environ["ROBOFLOW_API_KEY"])
project = rf.workspace("detrlane").project("lane-detection-awvt7")
version = project.version(2)
dataset = version.download("yolov8")
                