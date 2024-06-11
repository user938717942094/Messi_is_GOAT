import cv2
from ultralytics import YOLO

# YOLOv8 모델 로드
model = YOLO('yolov8n.pt')  # yolov8n.pt는 YOLOv8의 작은 모델, 다른 모델도 사용 가능
# model = YOLO(r'runs\detect\train37\weights\best.pt')

# 카메라 캡처 객체 생성 (웹캠 사용)
cap = cv2.VideoCapture(1,cv2.CAP_DSHOW) #윈도우즈는 Direct Show라는 아키텍쳐를 사용하기 때문에 이 객체를 명시해주는 것 입니다.
# cap.set(3, 640) ; cap.sep(4, 480) # 이 줄은 해상도를 고정하기 위한 줄입니다. 평소에는 필요 없습니다.

while True: # 이러면 openCV가 계속 프레임을 읽어서 영상을 실시간으로 처리하는 것처럼 보이게 합니다.
    # 프레임 읽기
    ret, frame = cap.read() #cap.read는 [프레임을 읽는데 성공여부 인 True/False의 값 (1번)] 과 [실제로 읽어온 프레임 (NumPy배열) (2번)] 처럼 2개의 인수를 반환합니다.
    #여기서 ret는 return의 약자로 만약 카메라 연결이 끊길 때 화면을 끄게 하면 ret같은 변수가 중요하겠지만 _같이 대충 만든 변수에 집어넣고 써먹지 않는다면 딱히 연결되든 아니든은 중요하지 않는것을 의미합니다.
    if not ret:
        break

    # OpenCV를 사용하여 읽어온 BGR 이미지를 RGB 이미지로 변환합니다.
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 객체 탐지 수행
    results = model(frame)

    # 객체 탐지 결과 확인
    if isinstance(results, list):
        results = results[0]

    # 바운딩 박스 추출
    boxes = results.xyxy[0].tensor.tolist()  # 객체 탐지 결과

    # 바운딩 박스 및 좌표 표시
    for box in boxes:
        x1, y1, x2, y2, conf, cls = box[:6]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        # 사각형 그리기
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # 좌표 점 및 텍스트 표시
        points = [(x1, y1), (x2, y2)]
        for (x, y) in points:
            cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)  # 빨간색 점 그리기
            cv2.putText(frame, f'({x}, {y})', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)  # 좌표 텍스트 표시

    # 결과 프레임 보여주기
    cv2.imshow('YOLOv8 Detection', frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 자원 해제
cap.release()
cv2.destroyAllWindows()