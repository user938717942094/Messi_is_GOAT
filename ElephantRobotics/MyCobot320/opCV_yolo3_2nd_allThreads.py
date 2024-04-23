import cv2
import numpy as np
import threading
import time

# YOLO 모델 설정
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers_indices = net.getUnconnectedOutLayers()

# output_layers_indices가 리스트가 아닌 경우에 대한 처리 추가
if isinstance(output_layers_indices, int):
    output_layers_indices = [output_layers_indices]

# output_layers_indices를 리스트 형태로 변환
output_layers = [layer_names[i - 1] for i in output_layers_indices]

class CameraThread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.cap = cv2.VideoCapture(1)  # 0은 기본 카메라를 의미합니다. 만약 다른 카메라를 사용하려면 해당 번호를 입력하세요.
        self.running = False

    def run(self):
        self.running = True
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break
            height, width, channels = frame.shape

            # 이미지 전처리
            blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
            net.setInput(blob)
            outs = net.forward(output_layers)

            # 감지된 사물 정보 저장
            class_ids = []
            confidences = []
            boxes = []
            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.5:
                        # Object detected
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)

                        # 사물의 경계 상자 좌표 계산
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)

                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

            # 경계 상자 그리기
            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
            font = cv2.FONT_HERSHEY_PLAIN
            colors = np.random.uniform(0, 255, size=(len(classes), 3))
            for i in range(len(boxes)):
                if i in indexes:
                    x, y, w, h = boxes[i]
                    label = str(classes[class_ids[i]])
                    color = colors[class_ids[i]]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(frame, label, (x, y + 30), font, 3, color, 3)

            # 결과 보여주기
            cv2.imshow("Image", frame)
            key = cv2.waitKey(1)
            if key == 27:
                break

        self.cap.release()
        cv2.destroyAllWindows()

    def stop(self):
        self.running = False

class OtherTaskThread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.counter = 0
        self.running = False

    def run(self):
        self.running = True
        while self.running:
            # 다른 작업 수행 (예: 단순한 카운터 증가)
            self.counter += 1
            print("Counter:", self.counter)
            # 작업을 위해 일시 중지하거나 다른 로직을 추가할 수 있습니다.
            time.sleep(1)

    def stop(self):
        self.running = False

camera_thread = CameraThread()
other_task_thread = OtherTaskThread()

# 스레드 시작
camera_thread.start()
other_task_thread.start()
