import time
import math

previous_positions = {}
previous_heights = {}
height_smooth = {}
v_approach_smooth = {}

def compute_motion(obj_id, cx, cy, height):

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

    # ----- Smooth height -----
    alpha_h = 0.7
    if obj_id in height_smooth:
        height_smooth[obj_id] = (
            alpha_h * height_smooth[obj_id]
            + (1 - alpha_h) * height
        )
    else:
        height_smooth[obj_id] = height

    smooth_height = height_smooth[obj_id]

    # ----- Raw approach velocity -----
    if obj_id in previous_heights:
        ph, pt = previous_heights[obj_id]
        dt = max(current_time - pt, 1e-5)
        v_approach_raw = (smooth_height - ph) / dt
    else:
        v_approach_raw = 0

    previous_heights[obj_id] = (smooth_height, current_time)

    # ----- Smooth approach velocity -----
    alpha_v = 0.8
    if obj_id in v_approach_smooth:
        v_approach_smooth[obj_id] = (
            alpha_v * v_approach_smooth[obj_id]
            + (1 - alpha_v) * v_approach_raw
        )
    else:
        v_approach_smooth[obj_id] = v_approach_raw

    v_approach = v_approach_smooth[obj_id]

    # ----- Clamp values -----
    v_approach = max(min(v_approach, 1000), -1000)

    return vx, vy, speed, v_approach