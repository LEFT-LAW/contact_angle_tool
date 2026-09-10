r"""Command line entry point for the contact-angle tool.

Single image (manual fallback) workflow::

    python main.py ..\contact_angle_tool\1.jpg

Auto-detect a single image::

    python main.py ..\contact_angle_tool\1.jpg --auto

Batch a whole folder (sorted by name, one tuning window per image)::

    python main.py ..\contact_angle_tool\batch

Results are written next to the input (``output/`` for a single image,
``<folder>/outputs/`` for a batch).
"""

from __future__ import annotations

import argparse
import os
import re
import sys

from contactangle import detect, imageio, manual, visualize
from contactangle.adjust import ResultAdjuster
from contactangle.angle import ALL_METHODS, compute_contact_angle

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Measure liquid-liquid contact angle vs. left wall.")
    p.add_argument("image", help="input image path, or a folder for batch processing")
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
    p.add_argument(
        "--outdir",
        default=None,
        help="output directory (default: 'output', or '<folder>/outputs' for a batch)",
    )
    p.add_argument("--name", default=None, help="output basename (default: input basename)")
    return p


def _natural_key(name: str):
    """Sort key so '2.jpg' comes before '10.jpg'."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", name)]


def acquire(img, auto: bool):
    """Return ``{'wall':..., 'interface':...}`` from auto-detect or manual pick."""
    if auto:
        found = detect.detect(img)
        if len(found["interface"]) >= 3:
            print(
                f"  auto: left wall x={found['wall'][0][0]:.0f}->{found['wall'][1][0]:.0f}, "
                f"interface points={len(found['interface'])}"
            )
            return {"wall": found["wall"], "interface": found["interface"]}
        print("  auto detection failed, falling back to manual picking.")
    return manual.run_manual(img)


def process_one(img, picked, args, outdir: str, name: str, image_path: str) -> bool:
    """Tune (optional) and save one image.  Returns True if saved."""
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
            return False
        method, window, result = tuned

    png_path = os.path.join(outdir, f"{name}_annotated.png")
    json_path = os.path.join(outdir, f"{name}.json")
    visualize.draw_result(img, wall, interface_points, result, out_path=png_path)
    visualize.save_json(json_path, image_path, wall, interface_points, result)
    print(
        f"  theta = {result.theta_deg:.2f} deg  "
        f"(method={method}, window={window:.2f}, exclude={result.exclude_px:.0f}px)"
    )
    print(f"  saved: {os.path.abspath(png_path)}")
    return True


def run_batch(folder: str, args) -> int:
    files = sorted(
        (f for f in os.listdir(folder) if os.path.splitext(f)[1].lower() in IMAGE_EXTS),
        key=_natural_key,
    )
    if not files:
        print(f"no images found in {folder}")
        return 1

    outdir = args.outdir or os.path.join(folder, "outputs")
    os.makedirs(outdir, exist_ok=True)
    print(f"batch: {len(files)} image(s) -> {os.path.abspath(outdir)}")

    for i, fn in enumerate(files, 1):
        print(f"[{i}/{len(files)}] {fn}")
        try:
            path = os.path.join(folder, fn)
            img = imageio.load_image(path, rotate=args.rotate)
            picked = acquire(img, auto=True)
            if not picked:
                print("  skipped (no selection).")
                continue
            name = os.path.splitext(fn)[0]
            if not process_one(img, picked, args, outdir, name, path):
                print("  skipped (cancelled).")
        except Exception as exc:  # keep the batch going
            print(f"  error: {exc}")
    print("batch done.")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if os.path.isdir(args.image):
        return run_batch(args.image, args)

    img = imageio.load_image(args.image, rotate=args.rotate)
    picked = acquire(img, auto=args.auto)
    if not picked:
        print("cancelled.")
        return 1

    outdir = args.outdir or "output"
    name = args.name or os.path.splitext(os.path.basename(args.image))[0]
    if not process_one(img, picked, args, outdir, name, args.image):
        print("cancelled at adjust step.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
