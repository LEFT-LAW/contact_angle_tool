"""Automatic detection of the left wall and the liquid-liquid interface.

Strategy
--------
* Walls: vertical-edge strength (Sobel-x) is projected over the container's
  vertical band; the strongest response in the left/right search zones gives a
  rough wall x, then each wall is refined per row and fitted with a line.
* Interface: the meniscus is a bright band with a strong dark->bright vertical
  gradient.  A dynamic-programming (Viterbi) tracker finds the smoothest path
  of maximum vertical gradient across the container, which is robust against
  isolated reflections.
"""

from __future__ import annotations

import cv2
import numpy as np


def _rough_walls(gray: np.ndarray) -> tuple[int, int]:
    h, w = gray.shape
    gx = np.abs(cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3))
    prof = gx[int(0.40 * h) : int(0.72 * h), :].sum(axis=0)

    def argmax(lo: float, hi: float) -> int:
        lo, hi = int(lo), int(hi)
        return lo + int(np.argmax(prof[lo:hi]))

    return argmax(0.28 * w, 0.43 * w), argmax(0.60 * w, 0.75 * w)


def _find_interface_band(gray: np.ndarray, x0: int, x1: int) -> tuple[int, int]:
    h = gray.shape[0]
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=5)
    # The liquid-liquid interface is always below the oil-air surface, so search
    # only the lower-middle part of the cell to avoid locking onto the oil top.
    lo, hi = int(0.46 * h), int(0.70 * h)
    rows = np.clip(gy[lo:hi, x0:x1], 0, None).sum(axis=1)
    peak = lo + int(np.argmax(rows))
    margin = int(0.08 * h)
    return max(0, peak - margin), min(h, peak + margin)


def _trace_interface(
    gray: np.ndarray, x0: int, x1: int, y0: int, y1: int, max_step: int = 4
) -> np.ndarray:
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=5)
    score = np.clip(gy[y0:y1, x0 + 5 : x1 - 5], 0, None)
    ny, nx = score.shape
    if ny == 0 or nx == 0:
        return np.empty((0, 2), dtype=float)

    dp = score.copy()
    back = np.zeros((ny, nx), dtype=np.int16)
    rows = np.arange(ny)
    for i in range(1, nx):
        prev = dp[:, i - 1]
        best = np.full(ny, -np.inf)
        best_dy = np.zeros(ny, dtype=np.int16)
        for dy in range(-max_step, max_step + 1):
            idx = rows + dy
            valid = (idx >= 0) & (idx < ny)
            cand = np.full(ny, -np.inf)
            cand[valid] = prev[idx[valid]]
            better = cand > best
            best[better] = cand[better]
            best_dy[better] = dy
        dp[:, i] = score[:, i] + best
        back[:, i] = best_dy

    y = int(np.argmax(dp[:, -1]))
    path = np.zeros(nx, dtype=int)
    for i in range(nx - 1, -1, -1):
        path[i] = y
        if i > 0:
            y += int(back[y, i])
    xs = np.arange(x0 + 5, x1 - 5)
    return np.column_stack([xs, y0 + path]).astype(float)


def _refine_wall(
    gray: np.ndarray, xc: int, ymid: int, band: int = 300, win: int = 18
) -> np.ndarray:
    """Fit a line to a wall edge around ``xc``; returns two points (top, bottom)."""
    h = gray.shape[0]
    y0, y1 = max(0, ymid - band), min(h, ymid + band)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    ys, xs = [], []
    for y in range(y0, y1, 8):
        seg = np.abs(gx[y, xc - win : xc + win])
        k = int(np.argmax(seg))
        if seg[k] > 0:
            xs.append(xc - win + k)
            ys.append(y)
    if len(xs) < 2:
        return np.array([[xc, y0], [xc, y1]], dtype=float)
    a, b = np.polyfit(np.array(ys), np.array(xs), 1)
    return np.array([[np.polyval([a, b], y0), y0], [np.polyval([a, b], y1), y1]], dtype=float)


def detect(img: np.ndarray, interface_band: tuple[int, int] | None = None) -> dict:
    """Detect the left/right walls and the interface from an RGB image.

    Returns a dict with ``wall`` (left wall, two points), ``interface``
    (Nx2 points), ``left_x``, ``right_x``, ``band`` and ``right_wall``.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
    lx, rx = _rough_walls(gray)
    if interface_band is None:
        y0, y1 = _find_interface_band(gray, lx, rx)
    else:
        y0, y1 = interface_band
    curve = _trace_interface(gray, lx, rx, y0, y1)
    ymid = int(np.median(curve[:, 1])) if len(curve) else (y0 + y1) // 2
    left_wall = _refine_wall(gray, lx, ymid)
    right_wall = _refine_wall(gray, rx, ymid)
    return {
        "wall": left_wall,
        "right_wall": right_wall,
        "interface": curve,
        "left_x": float(lx),
        "right_x": float(rx),
        "band": (int(y0), int(y1)),
    }
