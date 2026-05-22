import cv2
import numpy as np

cap = cv2.VideoCapture("test_video.mp4")
#print(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#print(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

while True:
    success,image = cap.read()
    
    if not success:
        break
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_blur = cv2.GaussianBlur(image_gray, (5, 5), 0)
    image_canny = cv2.Canny(image_blur, 50, 150)
    pts = np.array([[0, 720], [1280, 720], [740, 500], [480, 500]])

    blank_mask = np.zeros_like(image_canny)
    cv2.fillPoly(blank_mask, [pts], (255, 255, 255))
    image_mask = cv2.bitwise_and(image_canny, blank_mask)


    cv2.imshow("Video", image_mask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

