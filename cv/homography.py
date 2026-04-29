"""Homography helpers for mapping camera pixels onto parking lot coordinates."""

from __future__ import annotations

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover - fallback math remains testable.
    cv2 = None


def compute_homography(src_points: np.ndarray, dst_points: np.ndarray) -> np.ndarray:
    """Compute a 3x3 perspective transform from four or more point pairs.

    OpenCV is used when available. The SVD fallback keeps unit tests and design
    validation working in environments where OpenCV wheels are not installed.
    """
    src = np.asarray(src_points, dtype=float)
    dst = np.asarray(dst_points, dtype=float)
    if src.shape != dst.shape or src.shape[0] < 4 or src.shape[1] != 2:
        raise ValueError("src_points and dst_points must be Nx2 arrays with at least four rows.")
    if cv2 is not None:
        matrix, _ = cv2.findHomography(src, dst)
        if matrix is None:
            raise ValueError("OpenCV could not compute a homography for the given points.")
        return matrix

    rows = []
    for (x, y), (u, v) in zip(src, dst, strict=True):
        rows.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        rows.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    _, _, vh = np.linalg.svd(np.asarray(rows))
    matrix = vh[-1].reshape(3, 3)
    return matrix / matrix[2, 2]


def warp_point(point: tuple[float, float], homography: np.ndarray) -> tuple[float, float]:
    """Project a single image point through a homography matrix."""
    vector = np.array([point[0], point[1], 1.0], dtype=float)
    projected = np.asarray(homography, dtype=float) @ vector
    if projected[2] == 0:
        raise ValueError("Homogeneous coordinate is zero; point cannot be projected.")
    projected = projected / projected[2]
    return float(projected[0]), float(projected[1])


def warp_image(image: np.ndarray, homography: np.ndarray, output_size: tuple[int, int]) -> np.ndarray:
    """Apply a perspective warp to an image using OpenCV."""
    if cv2 is None:
        raise RuntimeError("OpenCV is required for image warping.")
    return cv2.warpPerspective(image, homography, output_size)
