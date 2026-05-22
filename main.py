import cv2

cap = cv2.VideoCapture("test_video.mp4")

while True:
    success,image = cap.read()
    cv2.imshow("Video", image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

