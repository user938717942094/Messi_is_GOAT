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
                
                # Print the structure of results
                #print(results)
                annotated_frame_1 = results[0].plot() #이 부분이 바운딩 박스를 그리는 1줄입니다. 견본으로서 남겨놓는 것 입니다.
                # Visualize the results on the frame
                annotated_frame = frame.copy()  # Create a copy of the original frame

                # Assuming results[0].boxes.xyxy for bounding box coordinates
                for obj in results[0].boxes.xyxy:  # Iterate over detected objects
                    x1, y1, x2, y2 = int(obj[0]), int(obj[1]), int(obj[2]), int(obj[3])
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw a green rectangle around the object
                # Draw a green rectangle around the object
                    
                    # Calculate the center point of the bounding box
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    
                    # Draw a red dot at the center of the bounding box
                    cv2.circle(annotated_frame, (center_x, center_y), 5, (0, 0, 255), -1)
                    # Draw text with coordinates at the center of the bounding box
                    cv2.putText(annotated_frame, f"Coord: {center_x}, {center_y}", (center_x, center_y-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (51, 255, 200), 2)
                # Display the annotated frame
                #cv2.imshow("rect", rect)
                cv2.imshow("YOLOv8 Tracking", annotated_frame)
                cv2.imshow("YOLOv8 Object", annotated_frame_1)
                
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                break

        cap.release()
        cv2.destroyAllWindows()
