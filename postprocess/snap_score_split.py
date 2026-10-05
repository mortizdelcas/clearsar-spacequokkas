#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import zipfile
from pathlib import Path


def _to_corners(bbox):
    x, y, w, h = bbox
    return float(x), float(y), float(x + w), float(y + h)


def _to_xywh(x1, y1, x2, y2):
    return [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]


def snap_round_corners(preds):
    out = []
    for p in preds:
        x1, y1, x2, y2 = _to_corners(p["bbox"])
        np_ = dict(p)
        np_["bbox"] = _to_xywh(round(x1), round(y1), round(x2), round(y2))
        out.append(np_)
    return out


def snap_per_box_optimal_corners(preds):
    """For each box, pick the (floor/ceil)^4 corner combination that
    maximises IoU with the original sub-pixel box."""
    out = []
    for p in preds:
        x1, y1, x2, y2 = _to_corners(p["bbox"])
        ax1, bx1 = math.floor(x1), math.ceil(x1)
        ax2, bx2 = math.floor(x2), math.ceil(x2)
        ay1, by1 = math.floor(y1), math.ceil(y1)
        ay2, by2 = math.floor(y2), math.ceil(y2)
        orig_area = max((x2 - x1) * (y2 - y1), 1e-9)
        cands = []
        for X1 in (ax1, bx1):
            for X2 in (ax2, bx2):
                if X2 <= X1:
                    continue
                for Y1 in (ay1, by1):
                    for Y2 in (ay2, by2):
                        if Y2 <= Y1:
                            continue
                        ix1, iy1 = max(X1, x1), max(Y1, y1)
                        ix2, iy2 = min(X2, x2), min(Y2, y2)
                        iw = max(ix2 - ix1, 0.0)
                        ih = max(iy2 - iy1, 0.0)
                        inter = iw * ih
                        cand_area = (X2 - X1) * (Y2 - Y1)
                        union = orig_area + cand_area - inter
                        cands.append((inter / max(union, 1e-9),
                                       X1, Y1, X2, Y2))
        if not cands:
            continue
        cands.sort(reverse=True)
        _, X1, Y1, X2, Y2 = cands[0]
        np_ = dict(p)
        np_["bbox"] = _to_xywh(X1, Y1, X2, Y2)
        out.append(np_)
    return out


def snap_score_split(preds, T=0.5):
    """High-confidence boxes (score >= T) use per-box optimal-corner snap;
    low-confidence boxes use plain round()."""
    high = [p for p in preds if p["score"] >= T]
    low  = [p for p in preds if p["score"] <  T]
    return snap_per_box_optimal_corners(high) + snap_round_corners(low)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in",  dest="inp",  required=True)
    ap.add_argument("--out", dest="outp", required=True)
    ap.add_argument("--T", type=float, default=0.5,
                    help="Score threshold for the high/low split.")
    ap.add_argument("--zip", action="store_true",
                    help="Also write a same-name .zip alongside the .json.")
    args = ap.parse_args()

    preds = json.load(open(args.inp))
    print(f"Loaded {len(preds)} predictions from {args.inp}")
    n_high = sum(1 for p in preds if p["score"] >= args.T)
    n_low = len(preds) - n_high
    print(f"  T={args.T}  -> {n_high} high-conf (per_box_optimal), "
          f"{n_low} low-conf (round_corners)")

    snapped = snap_score_split(preds, T=args.T)
    n_int = sum(1 for r in snapped
                for v in r["bbox"] if abs(v - round(v)) < 1e-9)
    n_total = 4 * max(len(snapped), 1)
    print(f"Snapped: {len(snapped)} preds, "
          f"{n_int}/{n_total} bbox values are integer "
          f"({100 * n_int / n_total:.1f}%)")

    Path(os.path.dirname(args.outp) or ".").mkdir(parents=True, exist_ok=True)
    with open(args.outp, "w") as f:
        json.dump(snapped, f)
        f.write("\n")
    print(f"Wrote {args.outp} ({os.path.getsize(args.outp) / 1e6:.2f} MB)")

    if args.zip:
        zip_path = os.path.splitext(args.outp)[0] + ".zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(args.outp, os.path.basename(args.outp))
        print(f"Wrote {zip_path} ({os.path.getsize(zip_path) / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
