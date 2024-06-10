import cv2
import numpy as np

cap = cv2.VideoCapture(0)

def process_frame(frame):
    height, width, _ = frame.shape
    roi_height = int(height / 3)
    roi_top = height - roi_height
    crop_img = frame[roi_top:, :]

    line_positions = [width // 5 * i for i in range(1, 5)]
    for pos in line_positions:
        cv2.line(crop_img, (pos, 0), (pos, roi_height), (0, 255, 0), 2)

    # Convert BGR to HSV
    hsv = cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV)

    # Define range of black color in HSV
    lower_black = np.array([26, 40, 154])
    upper_black = np.array([40, 255, 244])

    # Threshold the HSV image to get only black colors
    #mask = cv2.inRange(hsv, lower_black, upper_black)

    # Display the mask only
    cv2.imshow('Mask', crop_img)
    
    
while(True):
    ret, frame = cap.read()
    process_frame(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()

