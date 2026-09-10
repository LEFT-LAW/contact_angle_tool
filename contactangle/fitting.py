"""Curve fitting and line/circle intersection utilities.

All coordinates are in image pixels ``(x, y)`` with ``y`` increasing
downwards (as returned by ``matplotlib``'s ``event.xdata / event.ydata``).
"""

from __future__ import annotations

import numpy as np


def fit_circle(points: np.ndarray) -> tuple[np.ndarray, float]:
    """Algebraic (Kasa) least-squares circle fit.

    Returns ``(center, radius)`` where ``center`` is ``np.array([cx, cy])``.
    """
    pts = np.asarray(points, dtype=float)
    x, y = pts[:, 0], pts[:, 1]
    a = np.column_stack([x, y, np.ones_like(x)])
    b = x * x + y * y
    sol, *_ = np.linalg.lstsq(a, b, rcond=None)
    cx, cy = sol[0] / 2.0, sol[1] / 2.0
    r = float(np.sqrt(max(sol[2] + cx * cx + cy * cy, 0.0)))
    return np.array([cx, cy]), r


def fit_line(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Total-least-squares line fit.

    Returns ``(mean_point, unit_direction)``.
    """
    pts = np.asarray(points, dtype=float)
    mean = pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts - mean, full_matrices=False)
    direction = vt[0]
    norm = np.linalg.norm(direction)
    if norm > 0:
        direction = direction / norm
    return mean, direction


def line_line_intersection(
    p1: np.ndarray, p2: np.ndarray, q1: np.ndarray, q2: np.ndarray
) -> np.ndarray | None:
    """Intersection of two infinite lines, or ``None`` if parallel."""
    p1, p2, q1, q2 = map(lambda p: np.asarray(p, dtype=float), (p1, p2, q1, q2))
    d1, d2 = p2 - p1, q2 - q1
    denom = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(denom) < 1e-12:
        return None
    t = ((q1[0] - p1[0]) * d2[1] - (q1[1] - p1[1]) * d2[0]) / denom
    return p1 + t * d1


def circle_line_intersections(
    center: np.ndarray, radius: float, p1: np.ndarray, p2: np.ndarray
) -> list[np.ndarray]:
    """Intersection points of a circle and the infinite line through p1,p2."""
    p1 = np.asarray(p1, dtype=float)
    d = np.asarray(p2, dtype=float) - p1
    f = p1 - np.asarray(center, dtype=float)
    a = float(d @ d)
    if a < 1e-12:
        return []
    b = 2.0 * float(f @ d)
    c = float(f @ f) - radius * radius
    disc = b * b - 4 * a * c
    if disc < 0:
        return []
    s = np.sqrt(disc)
    return [p1 + ((-b - s) / (2 * a)) * d, p1 + ((-b + s) / (2 * a)) * d]


def circle_arc_points(
    center: np.ndarray, radius: float, points: np.ndarray, n: int = 200
) -> np.ndarray:
    """Sample the circle arc that spans the angular range of ``points``."""
    pts = np.asarray(points, dtype=float)
    ang = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    lo, hi = float(ang.min()), float(ang.max())
    if hi - lo > np.pi:  # handle wrap-around
        ang = np.mod(ang, 2 * np.pi)
        lo, hi = float(ang.min()), float(ang.max())
    t = np.linspace(lo, hi, n)
    return np.column_stack(
        [center[0] + radius * np.cos(t), center[1] + radius * np.sin(t)]
    )


def point_line_distances(points: np.ndarray, wall: np.ndarray) -> np.ndarray:
    """Absolute perpendicular distance from each point to the wall line."""
    pts = np.asarray(points, dtype=float)
    a, b = np.asarray(wall[0], float), np.asarray(wall[1], float)
    d = b - a
    n = np.linalg.norm(d)
    if n == 0:
        return np.linalg.norm(pts - a, axis=1)
    # 2D cross product magnitude / |d|
    return np.abs((d[0] * (pts[:, 1] - a[1]) - (pts[:, 0] - a[0]) * d[1]) / n)


def select_near_wall(
    points: np.ndarray, wall: np.ndarray, window_frac: float
) -> np.ndarray:
    """Boolean mask of interface points within ``window_frac * span`` of the wall.

    ``span`` is the largest distance of any interface point to the wall, so the
    window is resolution independent.
    """
    dist = point_line_distances(points, wall)
    span = float(dist.max()) if dist.size else 0.0
    if span <= 0:
        return np.ones(len(points), dtype=bool)
    return dist <= max(float(window_frac), 1e-6) * span


def fit_local_polynomial(
    points: np.ndarray, wall: np.ndarray, window_frac: float = 0.3, degree: int = 2
) -> dict:
    """Fit a low-order polynomial to the near-wall interface and extrapolate.

    The polynomial is evaluated at the intersection with the wall line to give
    the triple point, and its derivative there gives the local tangent.  The
    parametrisation (``y = f(x)`` or ``x = f(y)``) is chosen from the point
    spread so steep interfaces are handled too.

    Returns a dict with keys ``point``, ``tangent``, ``coeffs``, ``param``,
    ``mask``, ``t_min``, ``t_max``.
    """
    pts = np.asarray(points, dtype=float)
    mask = select_near_wall(pts, wall, window_frac)
    min_pts = degree + 2  # a few extra points for stability
    if mask.sum() < min_pts:
        order = np.argsort(point_line_distances(pts, wall))
        mask = np.zeros(len(pts), dtype=bool)
        mask[order[:min_pts]] = True

    local = pts[mask]
    if np.ptp(local[:, 0]) >= np.ptp(local[:, 1]):
        param = "x"
        coeffs = np.polyfit(local[:, 0], local[:, 1], degree)
    else:
        param = "y"
        coeffs = np.polyfit(local[:, 1], local[:, 0], degree)
    poly = np.poly1d(coeffs)

    a, b = np.asarray(wall[0], float), np.asarray(wall[1], float)
    d = b - a
    x_t = np.poly1d([d[0], a[0]])
    y_t = np.poly1d([d[1], a[1]])
    equation = (poly(x_t) - y_t) if param == "x" else (x_t - poly(y_t))

    mean = local.mean(axis=0)
    real_roots = [r.real for r in equation.roots if abs(r.imag) < 1e-6]
    if real_roots:
        t = min(real_roots, key=lambda s: float(np.linalg.norm(a + s * d - mean)))
        point = a + t * d
    else:  # fall back to the nearest interface point
        point = local[np.argmin(point_line_distances(local, wall))]

    if param == "x":
        slope = float(np.polyval(np.polyder(poly), point[0]))
        tangent = np.array([1.0, slope])
        t_min, t_max = float(local[:, 0].min()), float(local[:, 0].max())
    else:
        dxdy = float(np.polyval(np.polyder(poly), point[1]))
        tangent = np.array([dxdy, 1.0])
        t_min, t_max = float(local[:, 1].min()), float(local[:, 1].max())

    norm = np.linalg.norm(tangent)
    if norm > 0:
        tangent = tangent / norm

    return {
        "point": point,
        "tangent": tangent,
        "coeffs": coeffs,
        "param": param,
        "mask": mask,
        "t_min": t_min,
        "t_max": t_max,
    }


def sample_poly_fit(fit: dict, n: int = 200) -> np.ndarray:
    """Sample the local polynomial fit for drawing."""
    poly = np.poly1d(fit["coeffs"])
    t = np.linspace(fit["t_min"], fit["t_max"], n)
    if fit["param"] == "x":
        return np.column_stack([t, np.polyval(poly, t)])
    return np.column_stack([np.polyval(poly, t), t])
