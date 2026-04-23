def get_risk_level(count, max_capacity=25):
    """
    Calculates risk level based on percentage of max capacity.
    - LOW: < 60%
    - MODERATE: 60% - 80%
    - HIGH ALERT: > 80%
    """
    if count < 0.6 * max_capacity:
        return "LOW"
    elif count <= 0.8 * max_capacity:
        return "MODERATE"
    else:
        return "HIGH ALERT"
