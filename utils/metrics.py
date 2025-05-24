import numpy as np

def compute_leaflet_metrics_from_landmarks(landmarks: dict, bezier_curves: list = None, control_points: list = None):
    """
    Compute leaflet-relevant aortic valve metrics using manual landmark input,
    optionally substituting Bezier approximations.

    Parameters:
    - landmarks: dict with named points and curves (see structure above)
    - bezier_curves: optional list of 6 (or 3 merged) Nx3 arrays

    Returns:
    - metrics: dict of computed values
    """

    metrics = {}

    # -------------------------------
    # Leaflet Curve Preparation
    # -------------------------------

    if bezier_curves:
        if len(bezier_curves) == 6:
            # Merge left/right bezier curves into full leaflet edges
            R_curve = np.append(bezier_curves[1], bezier_curves[0], axis=0)
            L_curve = np.append(bezier_curves[3], bezier_curves[2], axis=0)
            N_curve = np.append(bezier_curves[5], bezier_curves[4], axis=0)
        elif len(bezier_curves) == 3:
            R_curve, L_curve, N_curve = bezier_curves
        else:
            raise ValueError("Expected 3 or 6 bezier curves.")
    else:
        # Use manually labeled leaflet curves (CI = commissure-insertion)
        R_curve = landmarks["RCI"]
        L_curve = landmarks["LCI"]
        N_curve = landmarks["NCI"]

    # -------------------------------
    # Cusp insertion length
    # -------------------------------
    metrics["R_cusp_insertion"] = compute_curve_length(R_curve)
    metrics["L_cusp_insertion"] = compute_curve_length(L_curve)
    metrics["N_cusp_insertion"] = compute_curve_length(N_curve)

    metrics["R1_cusp_insertion"] = compute_curve_length(bezier_curves[1])
    metrics["R2_cusp_insertion"] = compute_curve_length(bezier_curves[0])
    metrics["L1_cusp_insertion"] = compute_curve_length(bezier_curves[3])
    metrics["L2_cusp_insertion"] = compute_curve_length(bezier_curves[2])
    metrics["N1_cusp_insertion"] = compute_curve_length(bezier_curves[5])
    metrics["N2_cusp_insertion"] = compute_curve_length(bezier_curves[4])

    metrics["R_symmetry_ratio"] = min(metrics["R1_cusp_insertion"], metrics["R2_cusp_insertion"]) / max(metrics["R1_cusp_insertion"], metrics["R2_cusp_insertion"])
    metrics["L_symmetry_ratio"] = min(metrics["L1_cusp_insertion"], metrics["L2_cusp_insertion"]) / max(metrics["L1_cusp_insertion"], metrics["L2_cusp_insertion"])
    metrics["N_symmetry_ratio"] = min(metrics["N1_cusp_insertion"], metrics["N2_cusp_insertion"]) / max(metrics["N1_cusp_insertion"], metrics["N2_cusp_insertion"])

    # -------------------------------
    # Belly Angles (midpoint curvature estimate)
    # -------------------------------
    metrics["R_belly_angle"] = compute_angle_between_three_points(landmarks['R'], control_points[1], control_points[0])
    metrics["L_belly_angle"] = compute_angle_between_three_points(landmarks['L'], control_points[3], control_points[2])
    metrics["N_belly_angle"] = compute_angle_between_three_points(landmarks['N'], control_points[5], control_points[4])

    # -------------------------------
    # Commissural Angles (interleaflet)
    # -------------------------------
    # Example: angle between RLC - R - RNC (in radians or degrees)
    metrics["RL_angle"] = compute_angle_between_three_points(landmarks["RLC"], control_points[0],  control_points[3])
    metrics["LN_angle"] = compute_angle_between_three_points(landmarks["LNC"], control_points[2],  control_points[5])
    metrics["NR_angle"] = compute_angle_between_three_points(landmarks["RNC"], control_points[4],  control_points[1])

    return metrics


def compute_curve_length(curve):
    """
    Compute the total arc length of a 3D curve.

    Parameters:
    - curve: (N, 3) array or list of 3D points.

    Returns:
    - length: float, total length of the curve.
    """
    curve = np.asarray(curve)
    if curve.shape[0] < 2:
        return 0.0  # Not enough points

    # Compute pairwise distances between consecutive points
    diffs = np.diff(curve, axis=0)
    segment_lengths = np.linalg.norm(diffs, axis=1)
    total_length = np.sum(segment_lengths)

    return total_length.tolist()

def compute_angle_between_three_points(origin, point1, point2):

    v1 = point1 - origin
    v2 = point2 - origin

    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)

    dot = np.clip(np.dot(v1, v2), -1.0, 1.0)
    return np.degrees(np.arccos(dot)).tolist()