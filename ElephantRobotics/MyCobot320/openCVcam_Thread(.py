import cv2
import numpy as np
import threading

class ObjectDetectionThread(threading.Thread):
    def __init__(self, net, output_layers, classes, frame):
        threading.Thread.__init__(self)
        self.net = net
        self.output_layers = output_layers
        self.classes = classes
        self.frame = frame
        self.height, self.width, _ = frame.shape

    def run(self):
        blob = cv2.dnn.blobFromImage(self.frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        self.net.setInput(blob)
        outs = self.net.forward(self.output_layers)

        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x = int(detection[0] * self.width)
                    center_y = int(detection[1] * self.height)
                    w = int(detection[2] * self.width)
                    h = int(detection[3] * self.height)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        font = cv2.FONT_HERSHEY_PLAIN
        colors = np.random.uniform(0, 255, size=(len(self.classes), 3))
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(self.classes[class_ids[i]])
                color = colors[class_ids[i]]
                cv2.rectangle(self.frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(self.frame, label, (x, y + 30), font, 3, color, 3)

        cv2.imshow("Image", self.frame)

def main():
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

    # 카메라 설정
    cap = cv2.VideoCapture(1)  # USB 카메라

    while True:
        ret, frame = cap.read()
        if not ret:
            print("카메라에서 프레임을 읽을 수 없습니다.")
            break

        # 객체 감지 스레드 시작
        detection_thread = ObjectDetectionThread(net, output_layers, classes, frame)
        detection_thread.start()

        # 종료 조건
        if cv2.waitKey(1) & 0xFF == 27:  # ESC 키 누르면 종료
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
