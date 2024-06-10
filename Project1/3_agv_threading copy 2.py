import cv2
import numpy as np
from pymycobot.myagv import MyAgv
import threading

agv = MyAgv("/dev/ttyAMA2", 115200)

def process_frame(frame):
    height, width, _ = frame.shape
    roi_height = int(height / 5)
    roi_top = height - roi_height
    roi = frame[roi_top:, :]

    line_positions = [width // 5 * i for i in range(1, 5)]
    for pos in line_positions:
        cv2.line(roi, (pos, 0), (pos, roi_height), (0, 255, 0), 2)

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 64, 71], dtype=np.uint8)
    upper_white = np.array([61, 255, 255], dtype=np.uint8)
    white_mask = cv2.inRange(hsv, lower_white, upper_white)

    

    contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) >= 1:
        max_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(roi, [max_contour], -1, (0, 255, 0), 2)

        M = cv2.moments(max_contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            
            segment_width = width // 5
            center_segment = segment_width * 2
            
            if cx < center_segment - segment_width:
                return "LEFT"
            elif cx > center_segment + segment_width:
                return "RIGHT"
            else:
                return "STRAIGHT"


    return None

def camera_thread():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera error")
            break

        result = process_frame(frame)
        if result:
            print(result)
            if result == "LEFT":
                threading.Timer(0.3, agv.counterclockwise_rotation, (10,)).start()
            elif result == "RIGHT":
                threading.Timer(0.3, agv.counterclockwise_rotation, (10,)).start()
            else:
                agv.go_ahead(10)

        cv2.imshow("Frame", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

camera_thread = threading.Thread(target=camera_thread)
camera_thread.start()

camera_thread.join()