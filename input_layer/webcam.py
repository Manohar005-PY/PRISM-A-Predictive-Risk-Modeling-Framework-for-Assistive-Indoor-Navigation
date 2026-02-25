import cv2
from ultralytics import YOLO
import time

model = YOLO("yolov8n.pt")
model.to("cuda")
cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

prev_time = 0

if not cap.isOpened():
    print("camara cannot be opened")

while True:
    ret,frame = cap.read()
    if not ret:
        print("Failed to grab the frame")
        break
    
    results = model(frame, verbose=False)
    annotated_frame = results[0].plot()
    curr_time = time.time()
    fps = 1/(curr_time - prev_time) if prev_time != 0 else 0
    prev_time = curr_time
    cv2.putText(annotated_frame,f"FPS:{int(fps)}",(10,30),
                cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
    
    cv2.imshow("Yolo web cam", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()