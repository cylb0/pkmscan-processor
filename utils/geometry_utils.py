import numpy as np
import cv2


def order_quad_points(pts: np.ndarray) -> np.ndarray:
    """
    Orders 4 coordinates in a consistent clockwise manner.

    Ordering is determined by the polar angle of each point relative
    to their collective centroid. In OpenCV's coordinate system (y-axis pointing downwards),
    this results in a clockwise ordering [TL, TR, BR, BL].

    Args:
        pts: a 4 points numpy array (y,x)

    Returns:
        A float32 numpy array of points ordered clockwise.
    """
    if pts.shape[0] != 4:
        raise ValueError("Input must contain 4 points")

    centroid = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - centroid[1], pts[:, 0] - centroid[0])
    return pts[np.argsort(angles)].astype(np.float32)


def get_perspective_matrix(pts: np.ndarray, width: int, height: int) -> np.ndarray:
    """
    Computes a 3x3 transformation matrix.

    Args:
        pts: 4 ordered points (TL, TR, BR, BL)
        width: target width of the perfect rectangle
        height: target height of the perfect rectangle

    Returns:
        The 3x3 transformation matrix.
    """
    dst = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )

    return cv2.getPerspectiveTransform(pts, dst)
