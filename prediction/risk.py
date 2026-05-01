def compute_risk(TTC, v_approach, D, obj_class=None):

    # Avoid infinity issues
    if TTC == float("inf"):
        return "SAFE", (0,255,0), 0

    # Normalize components
    ttc_factor = 1 / (TTC + 1e-6)
    speed_factor = min(v_approach / 100, 1)
    distance_factor = min(1 / (D + 1e-6), 1)

    # Base risk score
    risk_score = (
        0.6 * ttc_factor +
        0.3 * speed_factor +
        0.1 * distance_factor
    )

    # Optional: object importance
    if obj_class == "person":
        risk_score *= 1.3
    elif obj_class == "chair":
        risk_score *= 0.8

    # Classification
    if risk_score > 1.2:
        return "DANGER", (0,0,255), risk_score
    elif risk_score > 0.6:
        return "CAUTION", (0,165,255), risk_score
    else:
        return "SAFE", (0,255,0), risk_score