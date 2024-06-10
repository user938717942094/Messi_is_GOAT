import cv2
import numpy as np
from pymycobot.myagv import MyAgv
import asyncio

agv = MyAgv("/dev/ttyAMA2", 115200)

async def process_frame(frame):
    height, width, _ = frame.shape
    roi_height = int(height / 5)
    roi_top = height - roi_height
    roi = frame[roi_top:, :]

    line_positions = [width // 5 * i for i in range(1, 5)]
    for pos in line_positions:
        cv2.line(roi, (pos, 0), (pos, roi_height), (0, 255, 0), 2)

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 64, 71], dtype=np.uint8)
    upper_white = np.array([61, 255, 255], dtype=np.uint8)
    white_mask = cv2.inRange(hsv, lower_white, upper_white)

    contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) >= 1:
        max_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(roi, [max_contour], -1, (0, 255, 0), 2)

        M = cv2.moments(max_contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            
            segment_width = width // 5
            center_segment = segment_width * 2
            
            if cx < center_segment - segment_width:
                return "LEFT"
            elif cx > center_segment + segment_width:
                return "RIGHT"
            else:
                return "STRAIGHT"

    return "STRAIGHT"  # 결과가 없을 때도 "STRAIGHT"를 반환하도록 수정

async def process_frame(frame):
    # 프레임 처리 함수 내용은 여기에 추가

    async def camera_stream():
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Camera error")
                break

            result = await process_frame(frame)
            if result:
                print(result)
                if result == "LEFT":
                    asyncio.run_coroutine_threadsafe(agv.counterclockwise_rotation(10), asyncio.get_event_loop())
                elif result == "RIGHT":
                    asyncio.run_coroutine_threadsafe(agv.clockwise_rotation(10), asyncio.get_event_loop())
                else:
                    asyncio.run_coroutine_threadsafe(agv.go_ahead(10), asyncio.get_event_loop())

            cv2.imshow("Frame", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    asyncio.run(camera_stream())
