from ultralytics import YOLO
import cv2

cap = cv2.VideoCapture("test_video.mp4")
model = YOLO("checkpoints/best.pt")

while True:
    success, image = cap.read()
    if not success:
        break
    
    results = model(image)
    annotated_frame = results[0].plot()
    cv2.imshow("Video", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
    

