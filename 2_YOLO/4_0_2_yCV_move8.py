from ultralytics import YOLO
import cv2
from ultralytics.utils.plotting import Annotator  # ultralytics.yolo.utils.plotting is deprecated
import numpy as np

model = YOLO('yolov8n.pt')
cap = cv2.VideoCapture(0)
#cap.set(3, 640); cap.set(4, 480)

while True:
    _, img = cap.read()




    
    # BGR to RGB conversion is performed under the hood
    # see: https://github.com/ultralytics/ultralytics/issues/2575
    results = model.predict(img)
    #model.predict(img, verbose=False) 이러면 화면에 결과가 뜨지 않는다.


    
    # fig, axs = plt.subplots(1,2, figsize=(10, 6))
    # axs = axs.ravel()
    # plt.subplots_adjust(left=0.1,bottom=0.1, 
    #                     right=0.9, top=0.9, 
    #                     wspace=0.2, hspace=0.4)

    # fig.suptitle("images", fontsize=18, y=0.95)

    # for i, (r, im) in enumerate(zip(results, images)):

    #     image = cv2.imread('/dir/' + im)

    #     c = r.boxes.xywh.tolist()[0] # To get the coordinates.
    #     x, y, w, h = c[0], c[1], c[2], c[3] # x, y are the center coordinates.
        
    #     axs[i].imshow(image)
    #     axs[i].add_patch(Rectangle((x-w/2, y-h/2), w, h,
    #                     edgecolor='blue', facecolor='none',
    #                     lw=3))

    for result in results:
        for box in result.boxes:
            left, top, right, bottom = int(obj[0]), int(obj[1]), int(obj[2]), int(obj[3])

            width = right - left
            height = bottom - top
            center = (left + int((right-left)/2), top + int((bottom-top)/2))
            label = results[0].names[int(box.cls)]
            confidence = float(box.conf.cpu())

            cv2.rectangle(img, (left, top),(right, bottom), (255, 0, 0), 2)

            cv2.putText(img, label,(left, bottom+20),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1, cv2.LINE_AA)

    # for r in results:
        
    # annotator = Annotator(img)
        
    #     boxes = r.boxes
    #     for box in boxes:
            
    #         b = box.xyxy[0]  # get box coordinates in (left, top, right, bottom) format
    #         c = box.cls
    #         annotator.box_label(b, model.names[int(c)])
          
    # img = annotator.result()  
    # print(results.[0])
    cv2.imshow('YOLO V8 Detection', img)

    # for result in results:
    #     # detection
    #     result.boxes.xyxy   # box with xyxy format, (N, 4)
    #     result.boxes.xywh   # box with xywh format, (N, 4)
    #     result.boxes.xyxyn  # box with xyxy format but normalized, (N, 4)
    #     result.boxes.xywhn  # box with xywh format but normalized, (N, 4)
    #     result.boxes.conf   # confidence score, (N, 1)
    #     result.boxes.cls    # cls, (N, 1)

    #     # segmentation
    #     #result.masks.masks     # masks, (N, H, W)
    #     #result.masks.segments  # bounding coordinates of masks, List[segment] * N

    #     # classification
    #     result.probs     # cls prob, (num_class, )
        
    if cv2.waitKey(1) & 0xFF == ord(' '):
        break

cap.release()
cv2.destroyAllWindows()