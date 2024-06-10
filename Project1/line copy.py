import cv2
import numpy as np
import threading
from pymycobot.myagv import MyAgv

agv = MyAgv("/dev/ttyACM0", 115200)

class CameraThread(threading.Thread):
    def __init__(self):
        super(CameraThread, self).__init__()
        self.cap = cv2.VideoCapture(0)
        self.running = False

    def run(self):
        self.running = True
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break
            action = process_frame(frame)
            perform_action(action)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        self.cap.release()
        cv2.destroyAllWindows()

def process_frame(frame):
    height, width, _ = frame.shape
    roi_height = int(height / 3)
    roi_top = height - roi_height
    crop_img = frame[roi_top:, :]

    line_positions = [width // 5 * i for i in range(1, 6)]
    for pos in line_positions:
        cv2.line(crop_img, (pos, 0), (pos, roi_height), (0, 255, 0), 2)

    # Convert BGR to HSV
    hsv = cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV)

    # Define range of black color in HSV
    lower_black = np.array([26, 40, 154])
    upper_black = np.array([40, 255, 244])

    # Threshold the HSV image to get only black colors
    mask = cv2.inRange(hsv, lower_black, upper_black)
    
    # Calculate the number of white pixels in each section
    section_width = width // 5
    section_counts = [np.count_nonzero(mask[:, i * section_width : (i + 1) * section_width]) for i in range(5)]

    # Determine action based on section counts
    if section_counts[0] + section_counts[1] > section_counts[3] + section_counts[4]:
        action = "LEFT"
    elif section_counts[0] + section_counts[1] < section_counts[3] + section_counts[4]:
        action = "RIGHT"
    else:
        action = "FORWARD"
    
    return action

def perform_action(action):
    if action == "LEFT":
        agv.counterclockwise_rotation(1)  # Rotate counterclockwise
    elif action == "RIGHT":
        agv.clockwise_rotation(1)  # Rotate clockwise
    else:
        agv.go_ahead(3)  # Move forward

if __name__ == "__main__":
    camera_thread = CameraThread()
    camera_thread.start()
