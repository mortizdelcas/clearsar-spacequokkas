import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from ensemble_boxes import weighted_boxes_fusion


# Order of the four ACTIVE sources in the per-record provenance arrays
# (cascade_TTA participates in WBF clustering at weight 0 but is omitted
# from provenance). Must match what the verifier checkpoints expect.
ACTIVE_SOURCES = ("cascade", "co-detr", "dfine", "dfine_TTA")


def group_by_image(dets):
    g = defaultdict(lambda: {"boxes": [], "scores": []})
    for d in dets:
        x, y, w, h = d["bbox"]
        g[d["image_id"]]["boxes"].append([x, y, x + w, y + h])
        g[d["image_id"]]["scores"].append(d["score"])
    for v in g.values():
        v["boxes"] = np.array(v["boxes"]) if v["boxes"] else np.zeros((0, 4))
        v["scores"] = np.array(v["scores"]) if v["scores"] else np.zeros(0)
    return g


def iou_xyxy_matrix(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ax1, ay1, ax2, ay2 = a[:, 0:1], a[:, 1:2], a[:, 2:3], a[:, 3:4]
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    ix1 = np.maximum(ax1, bx1); iy1 = np.maximum(ay1, by1)
    ix2 = np.minimum(ax2, bx2); iy2 = np.minimum(ay2, by2)
    iw = np.clip(ix2 - ix1, 0, None); ih = np.clip(iy2 - iy1, 0, None)
    inter = iw * ih
    aa = (ax2 - ax1) * (ay2 - ay1)
    bb = (bx2 - bx1) * (by2 - by1)
    union = aa + bb - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)


def fuse_with_provenance(grouped_5way, image_ids, image_sizes, weights_5,
                         iou_thr, match_iou, skip_box_thr=0.0001):
    PROV_IDX = (0, 2, 3, 4)  # cascade, co-detr, dfine, dfine_TTA
    out = []
    for img_id in image_ids:
        if img_id not in image_sizes:
            continue
        w_img, h_img = image_sizes[img_id]

        bl, sl, ll = [], [], []
        per_src_xyxy = []
        per_src_score = []
        for g in grouped_5way:
            if img_id in g and len(g[img_id]["boxes"]) > 0:
                bb = g[img_id]["boxes"].copy().astype(np.float64)
                bb_norm = bb.copy()
                bb_norm[:, [0, 2]] /= w_img
                bb_norm[:, [1, 3]] /= h_img
                bl.append(np.clip(bb_norm, 0, 1))
                sl.append(g[img_id]["scores"].astype(np.float64))
                ll.append(np.zeros(len(g[img_id]["scores"])))
                per_src_xyxy.append(bb)
                per_src_score.append(g[img_id]["scores"].astype(np.float64))
            else:
                bl.append(np.zeros((0, 4)))
                sl.append(np.array([]))
                ll.append(np.array([]))
                per_src_xyxy.append(np.zeros((0, 4)))
                per_src_score.append(np.zeros(0))

        fb, fs, _ = weighted_boxes_fusion(
            bl, sl, ll, weights=weights_5, iou_thr=iou_thr,
            skip_box_thr=skip_box_thr)
        if len(fb) == 0:
            continue

        valid = np.isfinite(fb).all(axis=1) & np.isfinite(fs)
        fb = fb[valid]
        fs = fs[valid]
        if len(fb) == 0:
            continue

        fb_abs = fb.copy()
        fb_abs[:, [0, 2]] *= w_img
        fb_abs[:, [1, 3]] *= h_img

        n_fused = len(fb_abs)
        records = [{
            "image_id": int(img_id),
            "category_id": 1,
            "bbox": [float(fb_abs[j, 0]), float(fb_abs[j, 1]),
                     float(fb_abs[j, 2] - fb_abs[j, 0]),
                     float(fb_abs[j, 3] - fb_abs[j, 1])],
            "score": float(fs[j]),
            "src_present": [0, 0, 0, 0],
            "src_score":   [0.0, 0.0, 0.0, 0.0],
            "src_max_iou": [0.0, 0.0, 0.0, 0.0],
        } for j in range(n_fused)]

        for prov_pos, s_idx in enumerate(PROV_IDX):
            iou_mat = iou_xyxy_matrix(fb_abs, per_src_xyxy[s_idx])
            if iou_mat.size == 0:
                continue
            src_max_iou = iou_mat.max(axis=1)
            masked = np.where(iou_mat >= match_iou,
                              per_src_score[s_idx][None, :], -1.0)
            src_max_score = masked.max(axis=1)
            src_max_score = np.where(src_max_score < 0, 0.0, src_max_score)
            for j in range(n_fused):
                records[j]["src_present"][prov_pos] = int(
                    src_max_iou[j] >= match_iou)
                records[j]["src_score"][prov_pos] = float(src_max_score[j])
                records[j]["src_max_iou"][prov_pos] = float(src_max_iou[j])

        out.extend(records)
    return out


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-cascade",     required=True)
    ap.add_argument("--pred-cascade-tta", required=True)
    ap.add_argument("--pred-co-detr",     required=True)
    ap.add_argument("--pred-dfine",       required=True)
    ap.add_argument("--pred-dfine-tta",   required=True)
    ap.add_argument("--ann", required=True)
    ap.add_argument("--out", required=True)

    ap.add_argument("--thr-cascade",     type=float, default=0.18)
    ap.add_argument("--thr-cascade-tta", type=float, default=0.10)
    ap.add_argument("--thr-co-detr",     type=float, default=0.06)
    ap.add_argument("--thr-dfine",       type=float, default=0.28)
    ap.add_argument("--thr-dfine-tta",   type=float, default=0.28)

    ap.add_argument("--w-cascade",     type=float, default=1.0)
    ap.add_argument("--w-cascade-tta", type=float, default=0.0)
    ap.add_argument("--w-co-detr",     type=float, default=1.0)
    ap.add_argument("--w-dfine",       type=float, default=1.2)
    ap.add_argument("--w-dfine-tta",   type=float, default=1.0)

    ap.add_argument("--iou",       type=float, default=0.75)
    ap.add_argument("--match-iou", type=float, default=0.55)
    return ap.parse_args()


def main():
    args = parse_args()

    print("Loading 5 source predictions ...")
    p_casc  = json.load(open(args.pred_cascade))
    p_cascT = json.load(open(args.pred_cascade_tta))
    p_codet = json.load(open(args.pred_co_detr))
    p_dfine = json.load(open(args.pred_dfine))
    p_dfineT = json.load(open(args.pred_dfine_tta))
    print(f"  cascade     = {len(p_casc):>7}    {args.pred_cascade}")
    print(f"  cascade_TTA = {len(p_cascT):>7}   {args.pred_cascade_tta}")
    print(f"  co-detr     = {len(p_codet):>7}    {args.pred_co_detr}")
    print(f"  dfine       = {len(p_dfine):>7}    {args.pred_dfine}")
    print(f"  dfine_TTA   = {len(p_dfineT):>7}   {args.pred_dfine_tta}")

    print("Pre-filtering by per-source thresholds ...")
    p_casc_f  = [d for d in p_casc  if d["score"] >= args.thr_cascade]
    p_cascT_f = [d for d in p_cascT if d["score"] >= args.thr_cascade_tta]
    p_codet_f = [d for d in p_codet if d["score"] >= args.thr_co_detr]
    p_dfine_f = [d for d in p_dfine if d["score"] >= args.thr_dfine]
    p_dfineT_f = [d for d in p_dfineT if d["score"] >= args.thr_dfine_tta]
    for n, k, t in (("cascade",     p_casc_f,  args.thr_cascade),
                    ("cascade_TTA", p_cascT_f, args.thr_cascade_tta),
                    ("co-detr",     p_codet_f, args.thr_co_detr),
                    ("dfine",       p_dfine_f, args.thr_dfine),
                    ("dfine_TTA",   p_dfineT_f, args.thr_dfine_tta)):
        print(f"  thr={t:.2f} -> {n:11s} = {len(k)}")

    print("Loading image sizes ...")
    coco = json.load(open(args.ann))
    sizes = {im["id"]: (im["width"], im["height"]) for im in coco["images"]}
    image_ids = sorted(sizes.keys())
    print(f"  {len(image_ids)} images")

    weights_5 = [args.w_cascade, args.w_cascade_tta, args.w_co_detr,
                 args.w_dfine, args.w_dfine_tta]
    print(f"\n5-way WBF: iou={args.iou}  weights={weights_5}  "
          f"match_iou={args.match_iou}")

    fused = fuse_with_provenance(
        [group_by_image(p_casc_f), group_by_image(p_cascT_f),
         group_by_image(p_codet_f), group_by_image(p_dfine_f),
         group_by_image(p_dfineT_f)],
        image_ids, sizes, weights_5,
        args.iou, args.match_iou)
    print(f"Fused: {len(fused)}")

    src_present_arr = np.array([f["src_present"] for f in fused])
    counts = src_present_arr.sum(axis=1)
    print("\nProvenance distribution (#sources voting per fused box):")
    for k in range(0, 5):
        n = int((counts == k).sum())
        print(f"  {k} sources: {n:>6} ({100 * n / len(fused):.1f}%)")
    for s, name in enumerate(ACTIVE_SOURCES):
        n = int(src_present_arr[:, s].sum())
        print(f"  contains {name:11s}: {n:>6} ({100 * n / len(fused):.1f}%)")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(fused, f)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
