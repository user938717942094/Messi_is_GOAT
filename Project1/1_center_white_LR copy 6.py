import cv2
import numpy as np
from pymycobot.myagv import MyAgv
import threading

# Initialize MyAgv object
agv = MyAgv("/dev/ttyAMA2", 115200)

def process_frame(frame):
    height, width, _ = frame.shape
    roi_height = int(height / 5)
    roi_top = height - roi_height
    roi = frame[roi_top:, :]

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 64, 71], dtype=np.uint8)
    upper_white = np.array([61, 255, 255], dtype=np.uint8)
    white_mask = cv2.inRange(hsv, lower_white, upper_white)

    # Find contours in the white mask
    contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Calculate black pixel counts in each third of the image
    black_pixel_counts = [np.sum(white_mask[:, i * (width // 3): (i + 1) * (width // 3)]) for i in range(3)]
    max_black_pixels_index = np.argmax(black_pixel_counts)
    print("Screen with the most detected black pixels:", max_black_pixels_index)
    
    # Trigger actions based on the screen with the most black pixels
    if max_black_pixels_index == 0:
        threading.Timer(0.01, agv.counterclockwise_rotation, (1,)).start()
    elif max_black_pixels_index == 2:
        threading.Timer(0.01, agv.clockwise_rotation, (1,)).start()
    elif max_black_pixels_index == 1:
        threading.Timer(0.01, agv.go_ahead, (3,)).start()

def camera_thread():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera error")
            break

        process_frame(frame)
        
        cv2.imshow("Frame", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Create and start the camera thread
camera_thread = threading.Thread(target=camera_thread)
camera_thread.start()

# Wait for the camera thread to finish
camera_thread.join()
