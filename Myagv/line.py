import cv2
import numpy as np
import threading

# Define global variables for storing the counts of detected objects on each section of the screen
left_count = 0
straight_count = 0
right_count = 0

# Function to process each frame
def process_frame(frame):
    global left_count, straight_count, right_count
    
    height, width, _ = frame.shape
    roi_height = int(height / 3)
    roi_top = height - roi_height
    crop_img = frame[roi_top:, :]

    # Divide the frame into 5 sections
    section_width = width // 5
    section_positions = [section_width * i for i in range(1, 5)]

    # Reset counts for each section
    left_count = 0
    straight_count = 0
    right_count = 0

    for i, pos in enumerate(section_positions):
        # Crop each section
        section = crop_img[:, i * section_width:(i + 1) * section_width]

        # Convert BGR to HSV
        hsv = cv2.cvtColor(section, cv2.COLOR_BGR2HSV)

        # Define range of black color in HSV
        lower_black = np.array([26, 40, 154])
        upper_black = np.array([40, 255, 244])

        # Threshold the HSV image to get only black colors
        mask = cv2.inRange(hsv, lower_black, upper_black)

        # Count non-zero pixels in the mask
        count = np.count_nonzero(mask)

        # Increment counts based on section index
        if i < 1:  # Left section
            left_count += count
        elif i < 3:  # Straight section
            straight_count += count
        else:  # Right section
            right_count += count

    # Decide the direction based on counts
    if left_count > straight_count and left_count > right_count:
        print("Turn left")
    elif right_count > straight_count and right_count > left_count:
        print("Turn right")
    else:
        print("Go straight")

# Function to continuously read frames from the camera
def camera_thread():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        process_frame(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Start camera thread
camera_thread = threading.Thread(target=camera_thread)
camera_thread.start()
