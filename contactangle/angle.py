"""Contact-angle computation from a wall line and an interface curve."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import fitting


@dataclass
class ContactAngleResult:
    """Result of a contact-angle measurement."""

    theta_deg: float
    point: np.ndarray  # triple point (x, y)
    tangent: np.ndarray  # unit tangent of the interface, pointing into the liquid
    wall_dir: np.ndarray  # unit wall direction pointing into the liquid (downwards)
    method: str
    fit_type: str
    center: np.ndarray | None = None
    radius: float | None = None
    line_point: np.ndarray | None = None
    line_dir: np.ndarray | None = None
    poly_fit: dict | None = None
    used_mask: np.ndarray | None = None
    window: float | None = None
    meta: dict = field(default_factory=dict)


LOCAL_METHODS = ("local-quad", "local-line", "local-circle")
GLOBAL_METHODS = ("global-circle", "global-line")
ALL_METHODS = LOCAL_METHODS + GLOBAL_METHODS


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def _fit_models(points: np.ndarray):
    """Return line and circle fits with their geometric RMS residuals."""
    mean, direction = fitting.fit_line(points)
    normal = np.array([-direction[1], direction[0]])
    line_res = float(np.sqrt(np.mean(((points - mean) @ normal) ** 2)))

    center, radius = fitting.fit_circle(points)
    circle_res = float(
        np.sqrt(np.mean((np.linalg.norm(points - center, axis=1) - radius) ** 2))
    )
    return (mean, direction, line_res), (center, radius, circle_res)


def _circle_contact(center, radius, wall, points):
    intersections = fitting.circle_line_intersections(center, radius, wall[0], wall[1])
    if intersections:
        point = min(
            intersections, key=lambda q: float(np.min(np.linalg.norm(points - q, axis=1)))
        )
    else:
        point = points[np.argmin(fitting.point_line_distances(points, wall))]
    radial = point - center
    tangent = _unit(np.array([-radial[1], radial[0]]))
    return point, tangent


def _line_contact(line_point, line_dir, wall, points):
    point = fitting.line_line_intersection(
        wall[0], wall[1], line_point, line_point + line_dir
    )
    if point is None:
        point = points[np.argmin(fitting.point_line_distances(points, wall))]
    return point, _unit(line_dir)


def compute_contact_angle(
    wall: list | np.ndarray,
    interface_points: np.ndarray,
    method: str = "local-quad",
    window: float = 0.3,
) -> ContactAngleResult:
    """Measure the contact angle between the interface and the wall.

    The angle is measured through the lower (conductive-liquid) phase: it is
    the angle between the wall direction pointing downwards and the interface
    tangent pointing into the liquid bulk.  A flat interface gives 90 degrees;
    a wetting meniscus gives < 90; a bulging (non-wetting) one gives > 90.

    Parameters
    ----------
    wall:
        Two points ``[[x1, y1], [x2, y2]]`` defining the wall line.
    interface_points:
        Points along the liquid-liquid interface.
    method:
        One of ``local-quad`` (default), ``local-line``, ``local-circle``,
        ``global-circle``, ``global-line``.
    window:
        For the local methods, the fraction of the interface span (distance
        from the wall to the far end of the interface) used for the fit.
    """
    wall = np.asarray(wall, dtype=float)
    if wall.shape != (2, 2):
        raise ValueError("wall must be two points [[x1, y1], [x2, y2]]")
    pts = np.asarray(interface_points, dtype=float)
    if pts.ndim != 2 or pts.shape[0] < 3:
        raise ValueError("at least 3 interface points are required")
    if method not in ALL_METHODS:
        raise ValueError(f"unknown method: {method}")

    wall_dir = _unit(wall[1] - wall[0])
    if wall_dir[1] < 0:  # point downwards, into the conductive liquid
        wall_dir = -wall_dir
    interior = pts.mean(axis=0)

    center = radius = None
    line_point = line_dir = None
    poly_fit = None
    used_mask = None

    if method in ("local-quad", "local-line"):
        degree = 2 if method == "local-quad" else 1
        poly_fit = fitting.fit_local_polynomial(pts, wall, window, degree=degree)
        point = poly_fit["point"]
        tangent = poly_fit["tangent"]
        used_mask = poly_fit["mask"]
        fit_type = f"poly{degree}"
    elif method == "local-circle":
        used_mask = fitting.select_near_wall(pts, wall, window)
        local = pts[used_mask]
        if len(local) < 3:
            used_mask = np.ones(len(pts), dtype=bool)
            local = pts
        center, radius = fitting.fit_circle(local)
        point, tangent = _circle_contact(center, radius, wall, pts)
        fit_type = "circle"
    elif method == "global-circle":
        center, radius = fitting.fit_circle(pts)
        point, tangent = _circle_contact(center, radius, wall, pts)
        used_mask = np.ones(len(pts), dtype=bool)
        fit_type = "circle"
    else:  # global-line
        line_point, line_dir = fitting.fit_line(pts)
        point, tangent = _line_contact(line_point, line_dir, wall, pts)
        used_mask = np.ones(len(pts), dtype=bool)
        fit_type = "line"

    if np.dot(tangent, interior - point) < 0:
        tangent = -tangent

    cos_theta = float(np.clip(np.dot(wall_dir, tangent), -1.0, 1.0))
    theta = float(np.degrees(np.arccos(cos_theta)))

    return ContactAngleResult(
        theta_deg=theta,
        point=np.asarray(point, dtype=float),
        tangent=tangent,
        wall_dir=wall_dir,
        method=method,
        fit_type=fit_type,
        center=center,
        radius=radius,
        line_point=line_point,
        line_dir=line_dir,
        poly_fit=poly_fit,
        used_mask=used_mask,
        window=window,
    )
