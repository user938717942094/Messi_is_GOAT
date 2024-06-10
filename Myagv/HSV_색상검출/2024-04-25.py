import cv2
import numpy as np

# 웹캠에서 영상 캡처
cap = cv2.VideoCapture(1)

while True:
    # 프레임 읽기
    ret, frame = cap.read()
    
    if not ret:
        break

    # BGR에서 HSV로 변환
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 빨간색의 HSV 범위 정의
    lower_red = np.array([0, 100, 100])
    upper_red = np.array([10, 255, 255])

    # 노란색의 HSV 범위 정의
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])

    # 파란색의 HSV 범위 정의
    lower_blue = np.array([110, 100, 100])
    upper_blue = np.array([130, 255, 255])

    # 보라색의 HSV 범위 정의
    lower_purple = np.array([140, 100, 100])
    upper_purple = np.array([160, 255, 255])

    # 이미지에서 빨간색, 노란색, 파란색, 보라색 영역을 추출
    red_mask = cv2.inRange(hsv_frame, lower_red, upper_red)
    yellow_mask = cv2.inRange(hsv_frame, lower_yellow, upper_yellow)
    blue_mask = cv2.inRange(hsv_frame, lower_blue, upper_blue)
    purple_mask = cv2.inRange(hsv_frame, lower_purple, upper_purple)

    # 외곽선 검출
    contours_red, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_yellow, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_blue, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_purple, _ = cv2.findContours(purple_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 빨간색 외곽선 그리기
    for contour in contours_red:
        cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)

    # 노란색 외곽선 그리기
    for contour in contours_yellow:
        cv2.drawContours(frame, [contour], -1, (0, 0, 255), 2)
    
    # 파란색 외곽선 그리기
    for contour in contours_blue:
        cv2.drawContours(frame, [contour], -1, (255, 0, 0), 2)

    # 보라색 외곽선 그리기
    for contour in contours_purple:
        cv2.drawContours(frame, [contour], -1, (128, 0, 128), 2)

    # 결과 영상 출력
    cv2.imshow('Color Detection', frame)

    # 'q'를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 작업이 끝나면 웹캠 해제 및 창 닫기
cap.release()
cv2.destroyAllWindows()
