from ultralytics import YOLO

model = YOLO("/Users/connorcola/Desktop/Projects/lane-detection/runs/detect/runs/train/yolov8n-4/weights/last.pt")

results = model.train(data="/Users/connorcola/Desktop/Projects/lane-detection/Lane-Detection-2/data.yaml", epochs=43, imgsz=640, batch=4, device='mps', project='runs/train', name='yolov8n_43')

print(results)