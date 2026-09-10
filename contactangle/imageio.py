"""Image loading and orientation helpers."""

from __future__ import annotations

import cv2
import numpy as np


def load_image(path: str, rotate: str | int = "auto") -> np.ndarray:
    """Load an image as RGB and orient it upright.

    Parameters
    ----------
    path:
        Path to the image file.
    rotate:
        ``"auto"`` rotates a landscape image 90 degrees clockwise (the
        container ends up portrait).  Otherwise pass ``0``, ``90``, ``180``
        or ``270`` for an explicit clockwise rotation.
    """
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return orient(img, rotate)


def orient(img: np.ndarray, rotate: str | int = "auto") -> np.ndarray:
    if rotate == "auto":
        h, w = img.shape[:2]
        if w > h:
            return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        return img

    rotate = int(rotate) % 360
    if rotate == 0:
        return img
    if rotate == 90:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if rotate == 180:
        return cv2.rotate(img, cv2.ROTATE_180)
    if rotate == 270:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    raise ValueError(f"Unsupported rotation: {rotate}")


def save_image(path: str, img: np.ndarray) -> None:
    """Save an RGB (or BGR-marked) array to disk."""
    cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
