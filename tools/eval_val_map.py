#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import io
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--expected", type=float, default=None,
                    help="If set, compare mAP@[.5:.95] against this value.")
    ap.add_argument("--atol", type=float, default=1e-3)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    ann = Path(args.ann)
    pred = Path(args.pred)
    if not ann.exists():
        print(f"[ERROR] annotation file not found: {ann}")
        return 2
    if not pred.exists():
        print(f"[ERROR] prediction file not found: {pred}")
        return 2

    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError:
        print("[ERROR] pycocotools not installed (pip install pycocotools)")
        return 2

    sink = io.StringIO() if args.quiet else sys.stdout
    with contextlib.redirect_stdout(sink):
        coco_gt = COCO(str(ann))
        coco_dt = coco_gt.loadRes(str(pred))
        ev = COCOeval(coco_gt, coco_dt, "bbox")
        ev.evaluate()
        ev.accumulate()
        ev.summarize()

    map_5095 = float(ev.stats[0])
    map_50 = float(ev.stats[1])

    print("=" * 72)
    print(f"COCO eval: {pred}")
    print("=" * 72)
    print(f"  ann        : {ann}")
    print(f"  mAP@.5:.95 = {map_5095:.4f}"
          + (f"  (expected {args.expected:.4f})" if args.expected is not None else ""))
    print(f"  mAP@.5     = {map_50:.4f}")
    print()

    if args.expected is None:
        return 0
    if abs(map_5095 - args.expected) <= args.atol:
        print(f"[PASS] mAP within tolerance ({args.atol}).")
        return 0
    print(f"[FAIL] mAP differs from expected by "
          f"{abs(map_5095 - args.expected):.4f} (atol={args.atol}).")
    return 1


if __name__ == "__main__":
    sys.exit(main())
