import cv2

cap = cv2.VideoCapture("test_video.mp4")

while True:
    success,image = cap.read()
    
    if not success:
        break
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_blur = cv2.GaussianBlur(image_gray, (5, 5), 0)
    image_canny = cv2.Canny(image_blur, 50, 300)
    cv2.imshow("Video", image_canny)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

