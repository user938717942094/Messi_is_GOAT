import cv2
import numpy as np
from pymycobot.myagv import MyAgv
import threading

# MyAgv 占쏙옙체 占쏙옙占쏙옙
MA = MyAgv("/dev/ttyAMA2", 115200)

# 占쏙옙占쏙옙占?占쏙옙占쏙옙 찾占싣쇽옙 占쏙옙占쏙옙占싹댐옙 占쌉쇽옙
def follow_yellow_line():
    # 카占쌨띰옙 占쏙옙占쏙옙
    cap = cv2.VideoCapture(0)

    while True:
        # 占쏙옙占쏙옙占쏙옙 占싻깍옙
        ret, frame = cap.read()
        if not ret:
            print("Camera error")
            break

        # 占쏙옙占쏙옙占?占쏙옙占쏙옙 찾占싣쇽옙 占쏙옙占쏙옙 占쏙옙占쏙옙 占쏙옙占쏙옙
        direction = process_frame(frame)
        print("Direction:", direction)
        if direction == "STRAIGHT":
            MA.go_ahead(5)  # 3占쏙옙 占쏙옙占쏙옙 占쏙옙占쏙옙
        elif direction == "LEFT":
            MA.counterclockwise_rotation(2)  # 占쏙옙占쏙옙占쏙옙占쏙옙 회占쏙옙
        elif direction == "RIGHT":
            MA.clockwise_rotation(2)  # 占쏙옙占쏙옙占쏙옙占쏙옙占쏙옙 회占쏙옙

        # 占쏙옙占쏙옙 占쏙옙占쏙옙
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            MA.stop()
            break

    # 카占쌨띰옙 占쏙옙占쏙옙
    cap.release()
    cv2.destroyAllWindows()

def process_frame(frame):
    # 占싱뱄옙占쏙옙 크占쏙옙 占쏙옙占쏙옙占쏙옙占쏙옙
    height, width, _ = frame.shape

    # ROI(Region of Interest) 占쏙옙占쏙옙
    roi_height = int(height / 3)
    roi_top = height - roi_height
    roi = frame[roi_top:, :]
    cv2.imshow("roi", roi)
    # 占쏙옙占쏙옙占?占쏙옙占쏙옙 占쏙옙占쏙옙
    lower_yellow = np.array([20, 100, 100], dtype=np.uint8)
    upper_yellow = np.array([30, 255, 255], dtype=np.uint8)

    # HSV 占쏙옙 占쏙옙占쏙옙占쏙옙占쏙옙 占쏙옙환
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # 占쏙옙占쏙옙占?占쏙옙占쏙옙크 占쏙옙占쏙옙
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # 占쏙옙占쏙옙占?占쏙옙 Contour 찾占쏙옙
    contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 占쏙옙占쏙옙占?占쏙옙占쏙옙 占쏙옙占쏙옙 占쏙옙占?    if len(contours) == 0:
        return None

    # 占쏙옙占쏙옙 큰 Contour 占쏙옙占쏙옙
    max_contour = max(contours, key=cv2.contourArea)

    # Contour占쏙옙 占쌩쏙옙 찾占쏙옙
    M = cv2.moments(max_contour)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        center_line = width // 2

        # 占쌩쏙옙 占쏙옙표占쏙옙 占쏙옙占쏙옙占쏙옙占쏙옙 占쏙옙占쏙옙 占쏙옙占쏙옙
        if center_line - 50 <= cx <= center_line + 50:
            return "STRAIGHT"
        elif cx < center_line - 50:
            return "LEFT"
        elif cx > center_line + 50:
            return "RIGHT"

    return None

# 占쏙옙占쏙옙占쏙옙 占쏙옙占쏙옙求占?占쏙옙占쏙옙占쏙옙 占쏙옙占쏙옙
follow_yellow_line()
