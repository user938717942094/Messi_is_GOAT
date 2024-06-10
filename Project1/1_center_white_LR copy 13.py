import cv2
import numpy as np
import threading

# Global variable for direction
direction = ''

def process_frame(frame):
    global direction

    height, width, _ = frame.shape
    roi_height = int(height / 3)
    roi_top = height - roi_height
    crop_img = frame[roi_top:, :]

    line_positions = [width // 5 * i for i in range(1, 5)]

    # Convert BGR to HSV
    hsv = cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV)

    # Define range of black color in HSV
    lower_black = np.array([26, 40, 0])
    upper_black = np.array([40, 255, 244])

    # Threshold the HSV image to get only black colors
    mask = cv2.inRange(hsv, lower_black, upper_black)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize centroid coordinates
    cx_sum = 0
    total_contours = 0

    # Calculate centroid for each contour
    for cnt in contours:
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cx_sum += cx
            total_contours += 1

    # Calculate average centroid position
    if total_contours != 0:
        average_cx = cx_sum / total_contours
    else:
        average_cx = width // 2  # Default to the center if no contours found

    # Determine direction based on centroid position
    if average_cx < width // 5:
        direction = 'Left'
    elif average_cx < 2 * width // 5:
        direction = 'Forward'
    elif average_cx < 3 * width // 5:
        direction = 'Forward'
    elif average_cx < 4 * width // 5:
        direction = 'Forward'
    else:
        direction = 'Right'

    # Display direction on the frame
    cv2.putText(crop_img, direction, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the processed frame
    cv2.imshow('Processed Frame', crop_img)

def camera_thread():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        process_frame(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

# Start camera thread
camera_thread = threading.Thread(target=camera_thread)
camera_thread.start()
