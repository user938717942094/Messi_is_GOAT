import cv2
import numpy as np
import threading
from pymycobot.myagv import MyAgv

agv = MyAgv("/dev/ttyAMA2", 115200)
# Assuming 'agv' is your object handling AGV actions

cap = cv2.VideoCapture(0)

def process_frame(frame):
    height, width, _ = frame.shape
    roi_height = int(height / 3)
    roi_top = height - roi_height
    crop_img = frame[roi_top:, :]

    line_positions = [width // 3 * i for i in range(1, 3)]
    for pos in line_positions:
        cv2.line(crop_img, (pos, 0), (pos, roi_height), (0, 255, 0), 2)

    # Convert BGR to HSV
    hsv = cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV)

    # Define range of black color in HSV
    lower_black = np.array([0, 64, 71])
    upper_black = np.array([61, 255, 255])

    # Threshold the HSV image to get only black colors
    mask = cv2.inRange(hsv, lower_black, upper_black)

    # Display the mask only
    cv2.imshow('Mask', mask)
    
    # Count black pixels
    black_pixel_counts = [np.sum(mask[:, i * (width // 3): (i + 1) * (width // 3)]) for i in range(3)]
    max_black_pixels_index = np.argmax(black_pixel_counts)
    print("Screen with the most detected black pixels:", max_black_pixels_index)
    
    # Trigger actions based on the screen with the most black pixels
    if max_black_pixels_index == 0:
        threading.Timer(0.01, agv.counterclockwise_rotation, (5,)).start()
    elif max_black_pixels_index == 2:
        threading.Timer(0.01, agv.clockwise_rotation, (5,)).start()
    elif max_black_pixels_index == 1:
        threading.Timer(0.01, agv.go_ahead, (9,)).start()

while True:
    ret, frame = cap.read()
    process_frame(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
