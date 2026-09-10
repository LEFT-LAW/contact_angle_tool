"""Interactive result tuning: adjust the near-wall window and fit model.

Controls are available both from the keyboard and from on-screen widgets, so
the tool still works when a CJK input method swallows plain letter keys.

The refraction zone next to the wall (UV-glue seam) is removed automatically
via a fixed pixel margin (``exclude_px``), so there is nothing to tune for it.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider

from . import visualize
from .angle import ALL_METHODS, compute_contact_angle

METHOD_CYCLE = [
    "local-quad",
    "local-line",
    "local-circle",
    "global-circle",
    "global-line",
]


class ResultAdjuster:
    """Show the fitted result and let the user tune window/model in real time.

    On-screen: drag the *window* slider, click the *model* button to cycle the
    fit, click *Save* / *Cancel*.

    Keyboard: ``[`` / ``]`` window, ``1``..``5`` pick model, ``m`` cycle model,
    ``Enter`` save, ``Esc`` cancel.
    """

    def __init__(
        self,
        img: np.ndarray,
        wall: np.ndarray,
        interface_points: np.ndarray,
        method: str = "local-quad",
        window: float = 0.3,
        exclude_px: float = 90.0,
        step: float = 0.05,
    ):
        self.img = img
        self.wall = np.asarray(wall, dtype=float)
        self.points = np.asarray(interface_points, dtype=float)
        self.method = method if method in ALL_METHODS else "local-quad"
        self.window = float(window)
        self.exclude_px = float(exclude_px)
        self.step = step
        self.result = None
        self.accepted = False

        h, w = img.shape[:2]
        self._lims = (-0.02 * w, 1.02 * w, 1.02 * h, -0.02 * h)

        self.fig = plt.figure(figsize=(9, 10))
        try:
            self.fig.canvas.manager.set_window_title("contact angle - adjust")
        except Exception:
            pass
        self.ax = self.fig.add_axes([0.06, 0.24, 0.88, 0.72])
        self.ax.set_axis_off()

        ax_window = self.fig.add_axes([0.30, 0.165, 0.45, 0.03])
        self.window_slider = Slider(
            ax_window, "window", 0.05, 1.0, valinit=self.window, valstep=0.05
        )
        self.window_slider.on_changed(self._on_window_slider)

        ax_model = self.fig.add_axes([0.28, 0.095, 0.49, 0.05])
        self.model_btn = Button(ax_model, self._model_label())
        self.model_btn.on_clicked(self._cycle_method)

        ax_save = self.fig.add_axes([0.16, 0.025, 0.30, 0.05])
        ax_cancel = self.fig.add_axes([0.54, 0.025, 0.30, 0.05])
        self.save_btn = Button(ax_save, "Save (Enter)")
        self.cancel_btn = Button(ax_cancel, "Cancel (Esc)")
        self.save_btn.on_clicked(self._accept)
        self.cancel_btn.on_clicked(self._cancel)

        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self._update()

    # ------------------------------------------------------------------ state
    def _model_label(self) -> str:
        return f"model: {self.method}   (click to cycle)"

    def _set_window(self, value: float) -> None:
        value = round(float(value) / self.step) * self.step
        self.window = float(np.clip(value, 0.05, 1.0))
        self.window_slider.eventson = False
        self.window_slider.set_val(self.window)
        self.window_slider.eventson = True
        self._update()

    def _cycle_method(self, _event=None) -> None:
        i = METHOD_CYCLE.index(self.method) if self.method in METHOD_CYCLE else 0
        self.method = METHOD_CYCLE[(i + 1) % len(METHOD_CYCLE)]
        self.model_btn.label.set_text(self._model_label())
        self._update()

    def _accept(self, _event=None) -> None:
        self.accepted = True
        plt.close(self.fig)

    def _cancel(self, _event=None) -> None:
        self.accepted = False
        plt.close(self.fig)

    # ----------------------------------------------------------------- render
    def _update(self) -> None:
        self.ax.cla()
        self.ax.imshow(self.img)
        self.ax.set_axis_off()
        self.ax.set_xlim(self._lims[0], self._lims[1])
        self.ax.set_ylim(self._lims[2], self._lims[3])

        self.result = compute_contact_angle(
            self.wall,
            self.points,
            method=self.method,
            window=self.window,
            exclude_px=self.exclude_px,
        )
        visualize.plot_overlay(self.ax, self.wall, self.points, self.result)

        self.ax.text(
            0.02,
            0.98,
            f"contact angle = {self.result.theta_deg:.2f} deg\n"
            f"method = {self.method}   window = {self.window:.2f}   "
            f"exclude = {self.exclude_px:.0f} px",
            transform=self.ax.transAxes,
            va="top",
            ha="left",
            fontsize=13,
            color="black",
            bbox=dict(facecolor="white", alpha=0.85, edgecolor="gray"),
        )
        self.ax.legend(loc="lower right", fontsize=8)
        self.fig.canvas.draw_idle()

    # ------------------------------------------------------------------ input
    def _on_window_slider(self, value) -> None:
        self.window = float(np.clip(value, 0.05, 1.0))
        self._update()

    def _on_key(self, event) -> None:
        key = event.key
        if key in ("]", "="):
            self._set_window(self.window + self.step)
        elif key in ("[", "-"):
            self._set_window(self.window - self.step)
        elif key == "m":
            self._cycle_method()
        elif key in ("1", "2", "3", "4", "5"):
            self.method = METHOD_CYCLE[int(key) - 1]
            self.model_btn.label.set_text(self._model_label())
            self._update()
        elif key in ("enter", "return"):
            self._accept()
        elif key == "escape":
            self._cancel()

    def run(self):
        """Block until accepted/cancelled.  Returns ``(method, window, result)``."""
        plt.show()
        if self.accepted and self.result is not None:
            return self.method, self.window, self.result
        return None
