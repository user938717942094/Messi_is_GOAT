import cv2
import numpy as np
import threading

class DirectionDetector(threading.Thread):
    def __init__(self, cap):
        super(DirectionDetector, self).__init__()
        self.cap = cap
        self.running = True

    def stop(self):
        self.running = False

    def run(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break
            height, width, _ = frame.shape
            roi_height = int(height / 3)
            roi_top = height - roi_height
            crop_img = frame[roi_top:, :]

            line_positions = [width // 3 * i for i in range(1, 3)]
            for pos in line_positions:
                cv2.line(crop_img, (pos, 0), (pos, roi_height), (0, 255, 0), 2)

            # Convert BGR to HSV
            hsv = cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV)

            # Define range of black color in HSV
            lower_black = np.array([0, 0, 0])
            upper_black = np.array([180, 255, 30])

            # Threshold the HSV image to get only black colors
            mask = cv2.inRange(hsv, lower_black, upper_black)

            

          

class VideoStream(threading.Thread):
    def __init__(self):
        super(VideoStream, self).__init__()
        self.cap = cv2.VideoCapture(0)
        self.running = True

    def stop(self):
        self.running = False

    def run(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break
            cv2.imshow('Video', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

# 메인 스레드에서 실행
if __name__ == "__main__":
    # 비디오 스트림 스레드 시작
    video_stream = VideoStream()
    video_stream.start()

    # 방향 감지 스레드 시작
    direction_detector = DirectionDetector(video_stream.cap)
    direction_detector.start()

    # 스레드 종료 대기
    video_stream.join()
    direction_detector.stop()
    direction_detector.join()

    # Release the camera and close all OpenCV windows
    video_stream.cap.release()
    cv2.destroyAllWindows()
