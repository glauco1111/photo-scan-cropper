#!/usr/bin/env python3
"""
crop_scans.py - Auto-crop scanned photos by detecting scanner border gradient.

How it works:
  The scanner creates a brightness gradient at the edges (scanner glass is lighter
  than photo content). The script scans from each edge inward and finds where the
  per-row/column mean brightness stabilizes into stable content, then adds a safety
  margin. The bottom typically has a wider border (photo paper + scanner shadow), so
  it uses a larger default margin.

Usage:
  # Preview crop boxes on thumbnails (no files modified):
  python crop_scans.py -i "//NAS/Scanner" --preview

  # Crop all JPEGs from input to output folder:
  python crop_scans.py -i "//NAS/Scanner" -o "//NAS/Cropped"

  # Adjust margin if the default cuts too little or too much:
  python crop_scans.py -i "//NAS/Scanner" -o "//NAS/Cropped" --margin 20 --margin-bottom 50

  # Override specific sides with fixed pixel values:
  python crop_scans.py -i "//NAS/Scanner" -o "//NAS/Cropped" --margin-left 34 --margin-right 37 --margin-top 42 --margin-bottom 76
"""

import argparse
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def detect_margins(
    img_array,
    pct=0.65,
    margin_lr=12,
    margin_top=12,
    margin_bottom=35,
):
    """
    Detect crop margins by finding where brightness gradient stabilizes.

    Returns (left, top, right, bottom) in pixels.
    pct: fraction of total brightness drop that triggers the cut point (0–1).
    """
    gray = img_array.mean(axis=2)
    h, w = gray.shape
    n = min(100, w // 2, h // 2)

    def find_by_mean_drop(vals, mg):
        start = vals[0]
        mn = min(vals)
        drop = start - mn
        if drop < 3:
            return mg  # no meaningful gradient → use minimum margin
        thresh = start - pct * drop
        for i, v in enumerate(vals):
            if v <= thresh:
                return i + mg
        return mg  # transition not found → use minimum margin

    left   = find_by_mean_drop([gray[:, x].mean()       for x in range(n)], margin_lr)
    right  = find_by_mean_drop([gray[:, w-1-x].mean()   for x in range(n)], margin_lr)
    top    = find_by_mean_drop([gray[y, :].mean()        for y in range(n)], margin_top)
    bottom = find_by_mean_drop([gray[h-1-y, :].mean()   for y in range(n)], margin_bottom)

    return left, top, right, bottom


def safe_crop(left, top, right, bottom, w, h, max_fraction=0.20):
    """Clamp margins so we never remove more than max_fraction of each dimension."""
    left   = min(left,   int(w * max_fraction))
    right  = min(right,  int(w * max_fraction))
    top    = min(top,    int(h * max_fraction))
    bottom = min(bottom, int(h * max_fraction))
    return left, top, right, bottom


def process_folder(input_dir, output_dir, args):
    input_path = Path(input_dir)
    output_path = Path(output_dir) if output_dir else None

    jpegs = sorted(
        p for p in input_path.iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg")
        and not p.name.startswith("_preview_")
    )

    if not jpegs:
        print(f"No JPEG files found in: {input_dir}")
        return

    if output_path and not args.preview:
        output_path.mkdir(parents=True, exist_ok=True)

    for img_path in jpegs:
        print(f"\n{img_path.name}")
        try:
            img = Image.open(img_path).convert("RGB")
            img_array = np.array(img)
            w, h = img.size

            auto_l, auto_t, auto_r, auto_b = detect_margins(
                img_array,
                pct=args.pct,
                margin_lr=args.margin,
                margin_top=args.margin_top,
                margin_bottom=args.margin_bottom,
            )

            # Apply per-side overrides if specified
            left   = args.fixed_left   if args.fixed_left   is not None else auto_l
            top    = args.fixed_top    if args.fixed_top    is not None else auto_t
            right  = args.fixed_right  if args.fixed_right  is not None else auto_r
            bottom = args.fixed_bottom if args.fixed_bottom is not None else auto_b

            left, top, right, bottom = safe_crop(left, top, right, bottom, w, h)

            new_w = w - left - right
            new_h = h - top - bottom
            print(f"  Original : {w} x {h}")
            print(f"  Margins  : L={left}  T={top}  R={right}  B={bottom}")
            print(f"  Result   : {new_w} x {new_h}")

            if args.preview:
                scale = 0.15
                thumb = img.copy()
                thumb.thumbnail((int(w * scale), int(h * scale)), Image.LANCZOS)
                tw, th = thumb.size
                sx, sy = tw / w, th / h
                draw = ImageDraw.Draw(thumb)
                draw.rectangle(
                    [left * sx, top * sy, (w - right) * sx, (h - bottom) * sy],
                    outline="red", width=3,
                )
                preview_path = input_path / f"_preview_{img_path.stem}.jpg"
                thumb.save(preview_path, quality=85)
                print(f"  Preview  : {preview_path}")
            else:
                box = (left, top, w - right, h - bottom)
                img.crop(box).save(output_path / img_path.name, quality=95)
                print(f"  Saved    : {output_path / img_path.name}")

        except Exception as e:
            print(f"  ERROR: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Auto-crop scanned JPEGs by detecting scanner border gradient.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input",  "-i", required=True,
                        help="Input folder containing JPEGs")
    parser.add_argument("--output", "-o",
                        help="Output folder (required unless --preview)")
    parser.add_argument("--preview", action="store_true",
                        help="Save small thumbnails with red crop box instead of cropping")

    parser.add_argument("--pct", type=float, default=0.65, metavar="0.0-1.0",
                        help="Fraction of brightness drop that triggers the cut (default: 0.65)")
    parser.add_argument("--margin", type=int, default=12, metavar="PX",
                        help="Safety pixels added after transition for left/right (default: 12)")
    parser.add_argument("--margin-top", type=int, default=12, metavar="PX",
                        help="Safety pixels for top (default: 12)")
    parser.add_argument("--margin-bottom", type=int, default=35, metavar="PX",
                        help="Safety pixels for bottom — larger because scanner/print border "
                             "is typically wider at the bottom (default: 35)")

    parser.add_argument("--fixed-left",   type=int, default=None, metavar="PX",
                        help="Override: use this fixed pixel margin on the left")
    parser.add_argument("--fixed-right",  type=int, default=None, metavar="PX",
                        help="Override: use this fixed pixel margin on the right")
    parser.add_argument("--fixed-top",    type=int, default=None, metavar="PX",
                        help="Override: use this fixed pixel margin on the top")
    parser.add_argument("--fixed-bottom", type=int, default=None, metavar="PX",
                        help="Override: use this fixed pixel margin on the bottom")

    args = parser.parse_args()

    if not args.preview and not args.output:
        parser.error("--output / -o is required unless --preview is specified")

    process_folder(args.input, args.output, args)
    print("\nDone.")


if __name__ == "__main__":
    main()
