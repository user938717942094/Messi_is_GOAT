import cv2
import numpy as np
#이걸 가지고 있는 이유는 이 코드가 색상의 융합이 되기 때문에 가시적이라 보기 편하기 떄문이다.

def main():
    # 웹캠 캡처 객체 생성
    cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        print("카메라를 열 수 없습니다.")
        return

    while True:
        # 프레임을 계속해서 읽음
        ret, frame = cap.read()

        if not ret:
            print("프레임을 읽을 수 없습니다. 종료합니다.")
            break

        # BGR에서 HSV 색공간으로 변환
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 색상 범위 정의 (파란색, 녹색, 빨간색)
        color_ranges = {
            'blue': ([110, 50, 50], [130, 255, 255], (255, 0, 0)),
            'green': ([50, 50, 50], [70, 255, 255], (0, 255, 0)),
            'red': ([0, 70, 50], [10, 255, 255], (0, 0, 255))
        }

        # 각 색상을 감지하고 직사각형으로 표시
        for color, (lower, upper, rect_color) in color_ranges.items():
            lower = np.array(lower, dtype="uint8")
            upper = np.array(upper, dtype="uint8")

            # 현재 색상에 대한 마스크 생성
            mask = cv2.inRange(hsv, lower, upper)
            # 마스크에 대한 윤곽선 찾기
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # 윤곽선을 둘러싼 직사각형 그리기
            for contour in contours:
                if cv2.contourArea(contour) > 300:  # 작은 윤곽은 무시
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), rect_color, 2)

        # 결과 이미지 표시
        cv2.imshow('Webcam - Multi Color Detection', frame)

        # 'q' 키를 누르면 루프에서 빠져나옴
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 자원 해제
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()