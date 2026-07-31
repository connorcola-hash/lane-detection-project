if __name__ == "__main__":
	from ultralytics import YOLO

	model = YOLO("checkpoints/last.pt")

	results = model.train(data="combined_data.yaml", epochs=43, imgsz=640, batch=-1, device=0, project='runs/train', name='yolov8n_43')

	print(results)