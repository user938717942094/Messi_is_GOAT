import cv2

# 카메라 인덱스 확인
for i in range(10):  # 일반적으로 10개까지 확인합니다.
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"카메라 인덱스 {i}에 연결된 카메라가 발견되었습니다.")
        cap.release()
