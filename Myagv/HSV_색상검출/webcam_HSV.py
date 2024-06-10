import cv2
import numpy as np

def detect_yellow(image):
    # BGR 이미지를 HSV 색 공간으로 변환
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # 노란색의 HSV 범위 정의
    lower_yellow = np.array([0, 0, 0])
    upper_yellow = np.array([61, 41, 101])
    
    # 노란색 영역을 마스킹
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    # 마스킹된 이미지에서 노란색 물체 외곽선 검출
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 외곽선 그리기
    cv2.drawContours(image, contours, -1, (0, 255, 255), 2)
    
    return image

# 웹캠 캡처
cap = cv2.VideoCapture(0)

while True:
    # 프레임 읽기
    ret, frame = cap.read()
    if not ret:
        break

    # 노란색 검출 함수 적용
    yellow_detected_frame = detect_yellow(frame)

    # 결과 출력
    cv2.imshow('Yellow Detection', yellow_detected_frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 리소스 해제
cap.release()
cv2.destroyAllWindows()
