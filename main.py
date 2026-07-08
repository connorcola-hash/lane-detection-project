import cv2
import numpy as np
from collections import deque

cap = cv2.VideoCapture("test_video.mp4")

def average_lines(array):
    left_candidates = []
    right_candidates = [] 
    left_line = None
    right_line = None
    y_a = 720
    y_b = 550

    if array is not None:
        for line in array:
            x1, y1, x2, y2 = line[0]
            slope, intercept = np.polyfit([x1, x2], [y1, y2], 1)
            if (abs(slope) < 0.25):
                continue#skips
            elif ((slope > 0)):
                left_candidates.append((slope, intercept))
            elif ((slope < 0)):
                right_candidates.append((slope, intercept))
        
        if left_candidates:
            slope_l, intercept_l = np.average(left_candidates, axis=0)
            x_la = int((y_a - intercept_l) / slope_l)
            x_lb = int((y_b - intercept_l) / slope_l)
            left_line = (x_la, y_a, x_lb, y_b)
        if right_candidates:
            slope_r, intercept_r = np.average(right_candidates, axis=0)
            x_ra = int((y_a - intercept_r) / slope_r)
            x_rb = int((y_b - intercept_r) / slope_r)
            right_line = (x_ra, y_a, x_rb, y_b)
    return left_line, right_line

def average_deq(deque_a):
    if len(deque_a) == 0:
        return None
    else:
        vals = np.average(deque_a, axis=0)
    x_a1 = int(vals[0])
    y_a1 = int(vals[1])
    x_a2 = int(vals[2])
    y_a2 = int(vals[3])

    return x_a1, y_a1, x_a2, y_a2
#print(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#print(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
left_deq = deque([], maxlen=8)
right_deq = deque([], maxlen=8)
while True:
    success,image = cap.read()
    
    if not success:
        break
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_blur = cv2.GaussianBlur(image_gray, (5, 5), 0)
    image_canny = cv2.Canny(image_blur, 50, 80)
    #pts = np.array([[0, 650],[0, 700], [1280, 700], [1280,670],[690, 515], [530, 515]]) {Hexagonal Polygon} full street
    pts = np.array([[110,700],[1000, 700],[690,515],[530,515]]) #Smaller Trapezoidal Polygon(bordering lanes)
    blank_mask = np.zeros_like(image_canny)
    cv2.fillPoly(blank_mask, [pts], (255, 255, 255))
    image_mask = cv2.bitwise_and(image_canny, blank_mask)

    lines = cv2.HoughLinesP(image_mask, 1, np.pi/180, 5, 10, 50)

    #if lines is not None:
    #    for line in lines:
    #        x1, y1, x2, y2 = line[0]
    #       cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 5)
    

    left, right = average_lines(lines)
        
    if left is not None:
        left_deq.append(left)
        cv2.line(image, left[0:2], left[2:4], (0,0, 255), 3)
    elif left is None:
        left = average_deq(left_deq)
        if left is not None:
            cv2.line(image, left[0:2], left[2:4], (0,0, 255), 3)

    if right is not None:
        cv2.line(image, right[0:2], right[2:4], (0,0, 255), 3)
        right_deq.append(right)
    elif right is None:
        right = average_deq(right_deq)
        if right is not None:
            cv2.line(image, right[0:2], right[2:4], (0,0, 255), 3)

    cv2.imshow("Video", image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

