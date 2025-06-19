"""Matrix Utils."""

import numpy as np


def rot_y(angle: float) -> np.ndarray:
    """Return Clockwise Rotation Matrix for the y-axis.

    Args:
        angle (float): Rotation Angle in radians.

    Returns:
        np.ndarray: Rotation Matrix
    """
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def inverse_rigid_transformation(tr: np.ndarray) -> np.ndarray:
    """Find inverse of rigid transformation matrix.

    tr: [R | t]
    inv_tr: [R' | -R't]

    Args:
        tr (np.ndarray): Rigid transformation matrix

    Returns:
        np.ndarray: Inverse of rigid transformation matrix
    """
    inv_tr = np.zeros_like(tr)
    tr_t = np.transpose(tr[:3, :3])
    inv_tr[:3, :3] = tr_t
    inv_tr[0:3, 3] = np.dot(-tr_t, tr[:3, 3])
    return inv_tr
