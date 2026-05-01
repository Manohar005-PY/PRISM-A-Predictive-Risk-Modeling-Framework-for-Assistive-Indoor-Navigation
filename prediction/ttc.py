def compute_ttc(D, v_approach, vx, vy, cx, frame_width):

    MIN_APPROACH = 1.5
    MIN_VY = 2

    # ----- Direction check -----
    approaching = (v_approach > MIN_APPROACH)

    # ----- Path (center region) -----
    center_min = frame_width * 0.3
    center_max = frame_width * 0.7

    in_path = (cx > center_min) and (cx < center_max)

    # ----- TTC -----
    if approaching and in_path:
        return D / (v_approach + 1e-6)
    else:
        return float("inf")