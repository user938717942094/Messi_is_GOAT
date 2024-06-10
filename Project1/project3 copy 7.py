import cv2
import numpy as np
import threading
import time
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
                    agv.counterclockwise_rotation(4)
                elif result == "FORWARD":
                    agv.go_ahead(10)
                elif result == "RIGHT":
                    agv.clockwise_rotation(4)
                elif result == "kang":
                    agv.move_control(160,128,140)
                else:
                    agv.counterclockwise_rotation(1)

            cv2.imshow("Frame", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    def stop(self):
        self.stopped = True
        self.cap.release()
        cv2.destroyAllWindows()

def process_frame(frame):
    height, width, _ = frame.shape
    section_width = int(width / 5)  # Calculate the width of each section
    section_height = int(height / 6)
    roi_top = height - section_height
    crop_img = frame[roi_top:, :]

    # Draw vertical lines to divide the ROI into five sections
    for i in range(1, 6):
        cv2.line(crop_img, (section_width * i, 0), (section_width * i, section_height), (0, 255, 0), 2)

    hsv = cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV)

    lower_yellow = np.array([26,60,109], dtype=np.uint8)
    upper_yellow = np.array([50,255,251], dtype=np.uint8)

    lower_red = np.array([26,60,109], dtype=np.uint8)
    upper_red = np.array([50,255,251], dtype=np.uint8)

    lower_purple = np.array([111,53,68], dtype=np.uint8)
    upper_purple = np.array([135,154,143], dtype=np.uint8)

    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    red_mask = cv2.inRange(hsv, lower_red, upper_red)
    purple_mask = cv2.inRange(hsv, lower_purple, upper_purple)

    contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_red, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_purple, _ = cv2.findContours(purple_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Contour detection and processing
    if len(contours_purple) >= len(contours) and len(contours_purple) >= len(contours_red):
        return "kang"
    elif len(contours) >= 1:
        max_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(crop_img, [max_contour], -1, (0, 255, 0), 2)

        # Calculate the center of the object
        M = cv2.moments(max_contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])

            center_lines = [section_width * (2 * i + 1) // 2 for i in range(5)]  # Calculate center lines for each section
            if 0 <= cx < section_width:
                return "LEFT"
            elif section_width <= cx < 4 * section_width:
                return "FORWARD"
            elif 4 * section_width <= cx < width:
                return "RIGHT"
            else :
                return "kang"

    return None


camera_thread = CameraThread()
camera_thread.start()

# Wait for the camera thread to finish
camera_thread.join()

# Release resources
camera_thread.stop()