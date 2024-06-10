import cv2
import numpy as np
from pymycobot.myagv import MyAgv
import threading

agv = MyAgv("/dev/ttyAMA2", 115200)

def process_frame(frame):
    height, width, _ = frame.shape
    roi_height = int(height / 3)
    roi_top = height - roi_height
    roi = frame[roi_top:, :]

    cv2.line(roi, (width // 5, 0), (width // 5, roi_height), (0, 255, 0), 2)
    cv2.line(roi, (width * 2 // 5, 0), (width * 2 // 5, roi_height), (0, 255, 0), 2)
    cv2.line(roi, (width * 3 // 5, 0), (width * 3 // 5, roi_height), (0, 255, 0), 2)
    cv2.line(roi, (width * 4 // 5, 0), (width * 4 // 5, roi_height), (0, 255, 0), 2)

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lower_white = np.array([21, 41, 199], dtype=np.uint8)
    upper_white = np.array([58, 255, 255], dtype=np.uint8)
    white_mask = cv2.inRange(hsv, lower_white, upper_white)

    contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 각 영역의 검출된 윤곽선 수 계산
    num_contours = [0] * 5
    for contour in contours:
        cx = contour[:, 0, 0].mean()  # 윤곽선의 x 좌표 평균
        if cx < width // 5:
            num_contours[0] += 1
        elif cx < width * 2 // 5:
            num_contours[1] += 1
        elif cx < width * 3 // 5:
            num_contours[2] += 1
        elif cx < width * 4 // 5:
            num_contours[3] += 1
        else:
            num_contours[4] += 1

    # 가장 많은 윤곽선이 있는 영역의 인덱스 찾기
    max_index = num_contours.index(max(num_contours))

    # 윤곽선 표시
    for contour in contours:
        cv2.drawContours(roi, [contour], -1, (0, 255, 0), 2)

    # 움직임 결정
    if max_index == 2:
        return "FORWARD"
    elif max_index < 2:
        return "LEFT"
    else:
        return "RIGHT"

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
                threading.Timer(0.3, agv.counterclockwise_rotation, (10,)).start()
            elif result == "RIGHT":
                threading.Timer(0.3, agv.counterclockwise_rotation, (10,)).start()
            elif result == "FORWARD":
                agv.go_ahead(10)

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
