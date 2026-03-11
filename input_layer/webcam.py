import cv2
from ultralytics import YOLO
import time
import math

# Load YOLO model
model = YOLO("yolov8n.pt")
model.to("cuda")

# Open webcam
cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Store previous states
previous_positions = {}
previous_heights = {}
height_smooth = {}

prev_time = 0

if not cap.isOpened():
    print("Camera cannot be opened")

while True:

    ret, frame = cap.read()
    if not ret:
        break

    annotated_frame = frame.copy()

    # Run YOLO tracking
    results = model.track(frame, persist=True, verbose=False)

    if results[0].boxes.id is not None:

        boxes = results[0].boxes.xyxy
        ids = results[0].boxes.id

        for box, obj_id in zip(boxes, ids):

            obj_id = int(obj_id)

            x1, y1, x2, y2 = map(int, box)

            # Bounding box center
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            width = x2 - x1
            height = y2 - y1

            # Draw bounding box
            cv2.rectangle(annotated_frame,(x1,y1),(x2,y2),(0,255,0),2)

            # Draw centroid
            cv2.circle(annotated_frame,(cx,cy),5,(0,0,255),-1)

            current_time = time.time()

            # ----- Centroid velocity -----

            if obj_id in previous_positions:

                px, py, pt = previous_positions[obj_id]

                dt = max(current_time - pt, 1e-5)

                vx = (cx - px) / dt
                vy = (cy - py) / dt

            else:
                vx, vy = 0, 0

            previous_positions[obj_id] = (cx, cy, current_time)

            speed = math.sqrt(vx**2 + vy**2)

            # ----- Smooth height for approach velocity -----

            alpha = 0.7

            if obj_id in height_smooth:
                height_smooth[obj_id] = alpha * height_smooth[obj_id] + (1-alpha) * height
            else:
                height_smooth[obj_id] = height

            smooth_height = height_smooth[obj_id]

            # ----- Approach velocity -----

            if obj_id in previous_heights:

                ph, pt = previous_heights[obj_id]

                dt = max(current_time - pt, 1e-5)

                v_approach = (smooth_height - ph) / dt

            else:
                v_approach = 0

            previous_heights[obj_id] = (smooth_height, current_time)

            # ----- Motion arrow -----

            scale = 0.1

            end_x = int(cx + vx * scale)
            end_y = int(cy + vy * scale)

            cv2.arrowedLine(
                annotated_frame,
                (cx,cy),
                (end_x,end_y),
                (255,0,0),
                2
            )

            # Draw object ID
            cv2.putText(
                annotated_frame,
                f"ID {obj_id}",
                (x1,y1-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0,255,255),
                2
            )

            # Show lateral velocity
            cv2.putText(
                annotated_frame,
                f"V:{speed:.1f}",
                (cx+10,cy),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255,0,255),
                2
            )

            # Show approach velocity (only if meaningful)
            if abs(v_approach) > 1:

                cv2.putText(
                    annotated_frame,
                    f"A:{v_approach:.1f}",
                    (x1,y2+20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0,255,255),
                    2
                )

    # ----- FPS -----

    curr_time = time.time()
    fps = 1/(curr_time-prev_time) if prev_time != 0 else 0
    prev_time = curr_time

    cv2.putText(
        annotated_frame,
        f"FPS:{int(fps)}",
        (10,30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,255,0),
        2
    )

    cv2.imshow("YOLO Motion Tracking", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()