import cv2
import numpy as np
from pymycobot.myagv import MyAgv
import threading


agv = MyAgv("/dev/ttyAMA2", 115200)

def process_frame(frame):
    height, width, _ = frame.shape

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_white = np.array([26,40,154], dtype=np.uint8)
    upper_white = np.array([40,255,244], dtype=np.uint8)
    white_mask = cv2.inRange(hsv, lower_white, upper_white)

    contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) >= 1:
        max_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(frame, [max_contour], -1, (0, 255, 0), 2)

        M = cv2.moments(max_contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])

            center_line = width // 2
            five_division = width // 5  # 가운데 기준 5등분 지점

            if cx < center_line - five_division:
                return "LEFT"
            elif cx > center_line + five_division:
                return "RIGHT"
            else:
                return "FORWARD"  # 중앙에 있으면 직진

    return None

def camera_thread():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera error")
            break

        result = process_frame(frame)
        if result:
            print(result)
            if result == "LEFT":
                agv.counterclockwise_rotation(1)
            elif result == "RIGHT":
                agv.clockwise_rotation(1)
            elif result == "FORWARD":
                agv.go_ahead(1)

        cv2.imshow("Frame", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# 메인 스레드에서 카메라 스레드 실행
camera_thread = threading.Thread(target=camera_thread)
camera_thread.start()

# 카메라 스레드가 종료될 때까지 대기
camera_thread.join()
