"""Overlay drawing and result saving."""

from __future__ import annotations

import json
import os

import numpy as np
import matplotlib.pyplot as plt

from . import fitting
from .angle import ContactAngleResult


def _extend(p1, p2, amount=4000.0):
    p1, p2 = np.asarray(p1, float), np.asarray(p2, float)
    d = p2 - p1
    n = np.linalg.norm(d)
    if n == 0:
        return p1, p2
    d = d / n
    return p1 - d * amount, p2 + d * amount


def _draw_fit(ax, interface_points, result):
    if result.poly_fit is not None:
        curve = fitting.sample_poly_fit(result.poly_fit)
        ax.plot(curve[:, 0], curve[:, 1], "-", color="blue", lw=2, label="local fit")
    elif result.center is not None:
        arc = fitting.circle_arc_points(result.center, result.radius, interface_points)
        ax.plot(arc[:, 0], arc[:, 1], "-", color="blue", lw=2, label="fitted circle")
    elif result.line_point is not None:
        p, d = result.line_point, result.line_dir
        q1, q2 = p - d * 3000, p + d * 3000
        ax.plot([q1[0], q2[0]], [q1[1], q2[1]], "-", color="blue", lw=2, label="fitted line")


def plot_overlay(ax, wall, interface_points, result: ContactAngleResult) -> None:
    """Draw wall, interface points (used ones highlighted), fit, tangent, triple."""
    wall = np.asarray(wall, float)
    interface_points = np.asarray(interface_points, float)

    a, b = _extend(wall[0], wall[1], amount=float(np.hypot(*np.ptp(interface_points, axis=0))) + 1e3)
    ax.plot([a[0], b[0]], [a[1], b[1]], "-", color="red", lw=2, label="wall")

    mask = result.used_mask
    if mask is None:
        mask = np.ones(len(interface_points), dtype=bool)
    mask = np.asarray(mask, dtype=bool)

    # Points dropped by the near-wall exclusion band (e.g. a UV-glue seam).
    excluded = np.zeros(len(interface_points), dtype=bool)
    if result.exclude_px and result.exclude_px > 0:
        dist = fitting.point_line_distances(interface_points, wall)
        excluded = (~mask) & (dist < result.exclude_px)

    others = ~mask & ~excluded
    if excluded.any():
        ax.plot(
            interface_points[excluded, 0],
            interface_points[excluded, 1],
            ".",
            color="gray",
            ms=6,
            label="excluded near wall",
        )
    if others.any():
        ax.plot(
            interface_points[others, 0],
            interface_points[others, 1],
            ".",
            color="deepskyblue",
            ms=6,
            label="interface pts",
        )
    if mask.any():
        ax.plot(
            interface_points[mask, 0],
            interface_points[mask, 1],
            "o",
            color="orange",
            mec="black",
            mew=0.5,
            ms=7,
            label="points used in fit",
        )

    _draw_fit(ax, interface_points, result)

    P, t = result.point, result.tangent
    q1, q2 = P - t * 400, P + t * 400
    ax.plot([q1[0], q2[0]], [q1[1], q2[1]], "-", color="lime", lw=2, label="tangent")
    ax.plot(P[0], P[1], "o", color="yellow", mec="black", ms=9, label="triple point")


def draw_result(
    img: np.ndarray,
    wall: np.ndarray,
    interface_points: np.ndarray,
    result: ContactAngleResult,
    out_path: str | None = None,
    show: bool = False,
) -> str | None:
    h, w = img.shape[:2]
    fig, ax = plt.subplots(figsize=(9, 9 * h / w))
    ax.imshow(img)
    ax.set_axis_off()
    ax.set_xlim(-0.02 * w, 1.02 * w)
    ax.set_ylim(1.02 * h, -0.02 * h)

    plot_overlay(ax, wall, interface_points, result)

    ax.text(
        0.02,
        0.98,
        f"contact angle = {result.theta_deg:.2f} deg\n"
        f"method={result.method}  window={result.window}  exclude={result.exclude_px:.0f}px",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=13,
        color="black",
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="gray"),
    )
    ax.legend(loc="lower right", fontsize=9)

    if out_path:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return out_path


def save_json(
    path: str,
    image_path: str,
    wall: np.ndarray,
    interface_points: np.ndarray,
    result: ContactAngleResult,
) -> str:
    data = {
        "image": os.path.basename(image_path),
        "theta_deg": round(result.theta_deg, 3),
        "method": result.method,
        "fit_type": result.fit_type,
        "window": result.window,
        "exclude_px": result.exclude_px,
        "wall": np.asarray(wall, float).round(2).tolist(),
        "interface_points": np.asarray(interface_points, float).round(2).tolist(),
        "used_mask": None
        if result.used_mask is None
        else np.asarray(result.used_mask, bool).tolist(),
        "triple_point": np.asarray(result.point, float).round(2).tolist(),
        "tangent": np.asarray(result.tangent, float).round(6).tolist(),
        "wall_dir": np.asarray(result.wall_dir, float).round(6).tolist(),
        "circle_center": None
        if result.center is None
        else np.asarray(result.center, float).round(2).tolist(),
        "circle_radius": None if result.radius is None else round(float(result.radius), 2),
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)
    return path
