import cv2
from input_layer.webcam import get_camera
from perception.detector import load_model, detect
from perception.tracker import compute_motion
from depth.midas_depth import DepthEstimator
from prediction.ttc import compute_ttc
from prediction.risk import compute_risk
from utils.draw import draw_box, draw_text

cap = get_camera()
model = load_model()
depth_estimator = DepthEstimator()

while True:

    ret, frame = cap.read()
    if not ret:
        break

    depth_map = depth_estimator.compute(frame)

    results = detect(model, frame)

    if results[0].boxes.id is not None:

        boxes = results[0].boxes.xyxy
        ids = results[0].boxes.id

        for box, obj_id in zip(boxes, ids):

            obj_id = int(obj_id)
            x1, y1, x2, y2 = map(int, box)

            cx = int((x1+x2)/2)
            cy = int((y1+y2)/2)
            height = y2-y1

            vx, vy, speed, v_approach = compute_motion(obj_id, cx, cy, height)
            D = depth_estimator.get_distance((x1,y1,x2,y2))

            frame_width = frame.shape[1]
            TTC = compute_ttc(D,v_approach,vx,vy,cx,frame_width)

            risk, color, score = compute_risk (TTC,v_approach,D,obj_class=None)
            print(f"TTC:{TTC:.2f}  V:{v_approach:.2f}  D:{D:.2f}  SCORE:{score:.2f}")
            draw_box(frame, (x1,y1,x2,y2), color)
            draw_text(frame, f"{risk}", (x1,y1-10))

    cv2.imshow("PRISM", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()