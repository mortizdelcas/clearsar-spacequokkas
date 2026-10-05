#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--img-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--category", default="RFI")
    args = ap.parse_args()

    img_dir = Path(args.img_dir)
    if not img_dir.exists():
        raise SystemExit(f"--img-dir not found: {img_dir}")

    paths = sorted(p for p in img_dir.iterdir()
                   if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
    if not paths:
        raise SystemExit(f"no images found in {img_dir}")

    images = []
    for p in paths:
        try:
            iid = int(p.stem)
        except ValueError:
            raise SystemExit(
                f"image {p.name} does not have an integer stem; "
                f"image_id is derived from int(filename stem).")
        with Image.open(p) as im:
            w, h = im.size
        images.append({
            "id": iid,
            "width": int(w),
            "height": int(h),
            "file_name": p.name,
        })

    coco = {
        "images": images,
        "annotations": [],
        "categories": [{"id": 1, "name": args.category}],
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(coco, f)

    print(f"Wrote {out}  ({len(images)} images)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
