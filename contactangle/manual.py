"""Interactive manual point picking (fallback workflow).

Workflow
--------
1. Click two points on the left side wall, then press ``Enter``.
2. Click several points along the liquid-liquid interface, then press
   ``Enter`` to compute.

Left click adds a point, right click removes the last point, ``Esc`` cancels.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


class ManualPicker:
    def __init__(self, img: np.ndarray):
        self.img = img
        self.wall: list[tuple[float, float]] = []
        self.interface: list[tuple[float, float]] = []
        self.state = "wall"
        self.result: dict | None = None

        h, w = img.shape[:2]
        self.fig, self.ax = plt.subplots(figsize=(9 * w / max(h, w), 9))
        self.ax.imshow(img)
        self.ax.set_axis_off()
        try:
            self.fig.canvas.manager.set_window_title("contact angle - manual")
        except Exception:
            pass

        self._wall_artist, = self.ax.plot([], [], "o-", color="red", ms=7, lw=1.5)
        self._iface_artist, = self.ax.plot([], [], "o-", color="deepskyblue", ms=5, lw=1)
        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self._redraw()

    def _on_click(self, event) -> None:
        if event.inaxes is not self.ax or event.xdata is None:
            return
        p = (float(event.xdata), float(event.ydata))
        target = self.wall if self.state == "wall" else self.interface
        if event.button == 1:
            if self.state == "wall" and len(self.wall) >= 2:
                self.wall = [self.wall[1], p]  # replace oldest once two exist
            else:
                target.append(p)
        elif event.button == 3:
            if target:
                target.pop()
        self._redraw()

    def _on_key(self, event) -> None:
        if event.key in ("enter", "return"):
            if self.state == "wall" and len(self.wall) == 2:
                self.state = "interface"
            elif self.state == "interface" and len(self.interface) >= 3:
                self.result = {
                    "wall": list(self.wall),
                    "interface": list(self.interface),
                }
                plt.close(self.fig)
        elif event.key == "escape":
            self.result = None
            plt.close(self.fig)
        self._redraw()

    def _redraw(self) -> None:
        wall = np.asarray(self.wall, dtype=float)
        iface = np.asarray(self.interface, dtype=float)

        if wall.size:
            self._wall_artist.set_data(wall[:, 0], wall[:, 1])
        else:
            self._wall_artist.set_data([], [])
        if iface.size:
            self._iface_artist.set_data(iface[:, 0], iface[:, 1])
        else:
            self._iface_artist.set_data([], [])

        if self.state == "wall":
            title = (
                f"STEP 1/2  left wall: click 2 points  "
                f"({len(self.wall)}/2)   right-click=undo   Enter=next"
            )
        else:
            title = (
                f"STEP 2/2  interface: click points along the liquid-liquid "
                f"interface  ({len(self.interface)})   right-click=undo   Enter=done"
            )
        self.ax.set_title(title, fontsize=10)
        self.fig.canvas.draw_idle()


def run_manual(img: np.ndarray) -> dict | None:
    """Run the interactive picker; returns ``{'wall', 'interface'}`` or None."""
    picker = ManualPicker(img)
    plt.show()
    return picker.result
