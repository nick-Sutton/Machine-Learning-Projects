import cv2, cvzone, math
from ultralytics import YOLO

def main():
    cap = cv2.VideoCapture(0)
    cap.set(3,1280)
    cap.set(4,720)
    model = YOLO("../Object Detection/Yolo-Weights/yolov8n.pt")

    classNames = ['person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 
    'hot dog', 'pizza', 'donut', 'cake', 'chair', 'sofa', 'pottedplant', 'bed', 'diningtable', 'toilet', 
    'tvmonitor', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 
    'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

    while True:
        ret, frame = cap.read()
        results = model(frame, stream=True)

        for i in results:
            boxes = i.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                #print(x1, y1, x2, y2)
                #cv2.rectangle(frame, (x1,y1), (x2,y2), (255,0,255),3)

                w, h = x2-x1, y2-y1
                cvzone.cornerRect(frame, (x1,y1,w,h))
                conf = math.ceil(box.conf[0]*100)/100

                cls = int(box.cls[0])


                cvzone.putTextRect(frame,f"{classNames[cls]} {conf}",(max(0,x1), max(35,y1)), scale=1, thickness=1)
        cv2.imshow("frame", frame)

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()