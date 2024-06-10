import cv2
import numpy as np

cap = cv2.VideoCapture(0)

def process_frame(frame):
    height, width, _ = frame.shape
    section_width = int(width / 5)  # Calculate the width of each section
    section_height = int(height / 3)
    roi_top = height - section_height
    roi = frame[roi_top:, :]

    # Draw vertical lines to divide the ROI into five sections
    for i in range(1, 5):
        cv2.line(roi, (section_width * i, 0), (section_width * i, section_height), (0, 255, 0), 2)
    
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    lower_black = np.array([26,40,154], dtype=np.uint8)
    upper_black = np.array([40,255,244], dtype=np.uint8)  # 약간의 임계값 추가하여 노이즈 감소
    black_mask = cv2.inRange(hsv, lower_black, upper_black)


    # Convert to grayscale and apply thresholding
    #gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    #_, binary_image = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(black_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Contour detection and processing
    if len(contours) >= 1:
        max_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(roi, [max_contour], -1, (0, 255, 0), 2)

        # Calculate the center of the object
        M = cv2.moments(max_contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])

            center_lines = [section_width * (2 * i + 1) // 2 for i in range(5)]  # Calculate center lines for each section
            for i, center_line in enumerate(center_lines):
                if cx < center_line - 50:
                    return f"LEFT {i+1}"  # Return which section is left if detected
                elif cx > center_line + 50:
                    return f"RIGHT {i+1}"  # Return which section is right if detected

    return None

while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera error")
        break

    result = process_frame(frame)
    
    if result:
        print(result)

    cv2.imshow("Frame", frame)

    if cv2.waitKey(2000) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
