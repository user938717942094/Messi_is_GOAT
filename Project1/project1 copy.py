import cv2
import numpy as np
import threading
from pymycobot.myagv import MyAgv

agv = MyAgv("/dev/ttyAMA2", 115200)

class CameraThread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.cap = cv2.VideoCapture(0)
        self.stopped = False

    def run(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if not ret:
                print("Camera error")
                break

            result = process_frame(frame)
            if result:
                print(result)
                if result == "LEFT":
                    agv.counterclockwise_rotation(3)
                elif result == "FORWARD":
                    agv.go_ahead(8)
                elif result == "RIGHT":
                    agv.clockwise_rotation(3)

            cv2.imshow("Frame", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    def stop(self):
        self.stopped = True
        self.cap.release()
        cv2.destroyAllWindows()

def process_frame(frame):
    height, width, _ = frame.shape
    section_width = int(width / 9)  # Divide the width into nine sections
    section_height = int(height / 4)
    roi_top = height - section_height
    roi = frame[roi_top:, :]

    # Draw vertical lines to divide the ROI into nine sections
    for i in range(1, 9):
        cv2.line(roi, (section_width * i, 0), (section_width * i, section_height), (0, 255, 0), 2)

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    lower_black = np.array([26,40,154], dtype=np.uint8)
    upper_black = np.array([40,255,244], dtype=np.uint8)
    black_mask = cv2.inRange(hsv, lower_black, upper_black)

    contours, _ = cv2.findContours(black_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) >= 1:
        max_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(roi, [max_contour], -1, (0, 255, 0), 2)

        M = cv2.moments(max_contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])

            if 0 <= cx < section_width * 3:  # Sections 0 and 1
                return "LEFT"
            elif section_width * 3 <= cx < section_width * 8:  # Sections 2, 3, 4, 5, 6
                return "FORWARD"
            elif section_width * 8 <= cx < width:  # Sections 7 and 8
                return "RIGHT"

    return None

camera_thread = CameraThread()
camera_thread.start()

# Wait for the camera thread to finish
camera_thread.join()

# Release resources
camera_thread.stop()
