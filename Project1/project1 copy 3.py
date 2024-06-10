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
                    agv.counterclockwise_rotation(1)
                elif result == "FORWARD":
                    agv.go_ahead(6)
                elif result == "RIGHT":
                    agv.clockwise_rotation(1)
                else:
                    agv.retreat(1)

            cv2.imshow("Frame", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    def stop(self):
        self.stopped = True
        self.cap.release()
        cv2.destroyAllWindows()

def process_frame(frame):
    height, width, _ = frame.shape
    roi_height = int(height / 3)
    roi_top = height - roi_height
    roi = frame[roi_top:, :]

    line_positions = [width // 5 * i for i in range(1, 6)]

    actions = ["LEFT", "FORWARD", "FORWARD", "FORWARD", "FORWARD", "RIGHT"]
    result = None

    for pos in line_positions:
        cv2.line(roi, (pos, 0), (pos, roi_height), (0, 255, 0), 2)
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        

        lower_black = np.array([26,40,86], dtype=np.uint8)
        upper_black = np.array([40,255,255], dtype=np.uint8)
        black_mask = cv2.inRange(hsv, lower_black, upper_black)

        contours, _ = cv2.findContours(black_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) >= 1:
            max_contour = max(contours, key=cv2.contourArea)
            cv2.drawContours(roi, [max_contour], -1, (0, 255, 0), 2)

            # Calculate the center of the object
            M = cv2.moments(max_contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])

            if i == 0 and cx < roi.shape[1] // 2:
                result = actions[i]
            elif 0 < i < 5 and len(contours) > 0:
                result = actions[i]
            elif i == 5 and cx > roi.shape[1] // 2:
                result = actions[i]

    return result

camera_thread = CameraThread()
camera_thread.start()

# Wait for the camera thread to finish
camera_thread.join()

# Release resources
camera_thread.stop()
