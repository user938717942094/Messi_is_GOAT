
# [6] 학습한 모델 테스트 하기


import cv2
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO(r'runs\detect\train37\weights\best.pt')

# Try different video capture backends if necessary
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)  # Change the index or backend if necessary

while True:
    if not cap.isOpened(): #이것은 비디오 캡처 장치가 열려 있는 지 보는 것이고 안쪽의 success는 그 비디오 장치에서 프레임을 열었는지 보는 것입니다.
        #이중으로 감지 장치가 걸려있기에 print구문으로 사용자는 카메라 USB가 문제인지 카메라 모듈이 망가졌는지 등을 알 수 있습니다. 디버깅용입니다.
        print("Error: Could not open video device")
    else:
        # Loop through the video frames
        while cap.isOpened():
            # Read a frame from the video
            success, frame = cap.read()

            if success:
                # Run YOLOv8 tracking on the frame, persisting tracks between frames
                results = model.track(frame, persist=True)
                # Visualize the results on the frame
                annotated_frame = results[0].plot()
                # Display the annotated frame
                cv2.imshow("Paul", annotated_frame)
                # Break the loop if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                # Break the loop if the end of the video is reached or an error occurs
                break

        # Release the video capture object and close the display window
        cap.release()
        cv2.destroyAllWindows()


    # ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ
    # 여튼 이건 된다. 바운딩 박스 출력을 어떻게 하나 그게 궁금한거지.