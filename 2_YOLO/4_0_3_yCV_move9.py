import cv2
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO(r'runs\detect\train37\weights\best.pt')

# Try different video capture backends if necessary
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)  # Change the index or backend if necessary

while True:
    if not cap.isOpened():
        print("Error: Could not open video device")
    else:
        while cap.isOpened():
            success, frame = cap.read()

            if success:
                # Run YOLOv8 tracking on the frame, persisting tracks between frames
                results = model.track(frame, persist=True)
                
                # Visualize the results on the frame
                annotated_frame = frame.copy()  # Create a copy of the original frame
                
                # Draw bounding boxes on the annotated frame
                for obj in results.xyxy[0]:  # Iterate over detected objects
                    x1, y1, x2, y2 = int(obj[0]), int(obj[1]), int(obj[2]), int(obj[3])
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw a green rectangle around the object
                
                # Display the annotated frame
                cv2.imshow("YOLOv8 Tracking", annotated_frame)
                
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                break

        cap.release()
        cv2.destroyAllWindows()
