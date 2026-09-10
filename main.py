r"""Command line entry point for the contact-angle tool.

Manual (fallback) workflow::

    python main.py ..\contact_angle_tool\1.jpg

Click two points on the left wall (Enter), then points along the
liquid-liquid interface (Enter).  A tuning window then opens where the
near-wall fit window and model can be adjusted before saving.

Results are written to ``output/``.
"""

from __future__ import annotations

import argparse
import os
import sys

from contactangle import detect, imageio, manual, visualize
from contactangle.adjust import ResultAdjuster
from contactangle.angle import ALL_METHODS, compute_contact_angle


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Measure liquid-liquid contact angle vs. left wall.")
    p.add_argument("image", help="input image path")
    p.add_argument(
        "--rotate",
        default="auto",
        help="'auto' (default) or 0/90/180/270 clockwise",
    )
    p.add_argument(
        "--method",
        default="local-quad",
        choices=list(ALL_METHODS),
        help="interface fit method (default: local-quad)",
    )
    p.add_argument(
        "--window",
        type=float,
        default=0.3,
        help="near-wall window as a fraction of the interface span (default: 0.3)",
    )
    p.add_argument(
        "--exclude-px",
        type=float,
        default=90.0,
        help="pixels next to the left wall to ignore, i.e. the UV-glue "
        "refraction zone (default: 90)",
    )
    p.add_argument(
        "--auto",
        action="store_true",
        help="auto-detect the left wall and interface instead of picking manually",
    )
    p.add_argument(
        "--no-adjust",
        action="store_true",
        help="skip the interactive tuning window and save immediately",
    )
    p.add_argument("--outdir", default="output", help="output directory (default: output)")
    p.add_argument("--name", default=None, help="output basename (default: input basename)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    img = imageio.load_image(args.image, rotate=args.rotate)

    picked = None
    if args.auto:
        found = detect.detect(img)
        if len(found["interface"]) >= 3:
            picked = {"wall": found["wall"], "interface": found["interface"]}
            print(
                f"auto: left wall x={found['wall'][0][0]:.0f}->{found['wall'][1][0]:.0f}, "
                f"interface points={len(found['interface'])}"
            )
        else:
            print("auto detection failed, falling back to manual picking.")

    if picked is None:
        picked = manual.run_manual(img)
    if not picked:
        print("cancelled.")
        return 1

    wall = picked["wall"]
    interface_points = picked["interface"]

    method, window = args.method, args.window
    exclude_px = args.exclude_px
    if args.no_adjust:
        result = compute_contact_angle(
            wall, interface_points, method=method, window=window, exclude_px=exclude_px
        )
    else:
        tuned = ResultAdjuster(
            img, wall, interface_points, method=method, window=window, exclude_px=exclude_px
        ).run()
        if tuned is None:
            print("cancelled at adjust step.")
            return 1
        method, window, result = tuned

    name = args.name or os.path.splitext(os.path.basename(args.image))[0]
    png_path = os.path.join(args.outdir, f"{name}_annotated.png")
    json_path = os.path.join(args.outdir, f"{name}.json")

    visualize.draw_result(img, wall, interface_points, result, out_path=png_path)
    visualize.save_json(json_path, args.image, wall, interface_points, result)

    print(
        f"theta = {result.theta_deg:.2f} deg  "
        f"(method={method}, window={window:.2f}, exclude={result.exclude_px:.0f}px, "
        f"fit={result.fit_type})"
    )
    print(f"annotated: {os.path.abspath(png_path)}")
    print(f"json     : {os.path.abspath(json_path)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
