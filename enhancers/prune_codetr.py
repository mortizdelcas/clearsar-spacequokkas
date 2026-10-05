import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "verifier"))
from verifier_model import VerifierV2 # noqa
from verifier_features import iou_xywh, containment, make_crop # noqa


def iou_xyxy_vectorized(a_boxes, b_boxes):
    if len(a_boxes) == 0 or len(b_boxes) == 0:
        return np.zeros((len(a_boxes), len(b_boxes)))
    a = np.array([[b[0], b[1], b[0]+b[2], b[1]+b[3]] for b in a_boxes])
    b = np.array([[b[0], b[1], b[0]+b[2], b[1]+b[3]] for b in b_boxes])
    ix1 = np.maximum(a[:, 0:1], b[:, 0])
    iy1 = np.maximum(a[:, 1:2], b[:, 1])
    ix2 = np.minimum(a[:, 2:3], b[:, 2])
    iy2 = np.minimum(a[:, 3:4], b[:, 3])
    iw = np.clip(ix2 - ix1, 0, None)
    ih = np.clip(iy2 - iy1, 0, None)
    inter = iw * ih
    aa = (a[:, 2]-a[:, 0]) * (a[:, 3]-a[:, 1])
    bb = (b[:, 2]-b[:, 0]) * (b[:, 3]-b[:, 1])
    union = aa[:, None] + bb[None, :] - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)


def neighborhood_features_flat(preds):
    n = len(preds)
    if n == 0:
        return []
    scores = [p["score"] for p in preds]
    order = sorted(range(n), key=lambda i: -scores[i])
    rank = {idx: r for r, idx in enumerate(order)}
    results = []
    for i in range(n):
        count_nbr = 0
        max_iou_h = 0.0
        max_cont_h = 0.0
        for j in range(n):
            if i == j:
                continue
            v = iou_xywh(preds[i]["bbox"], preds[j]["bbox"])
            if v >= 0.5:
                count_nbr += 1
            if rank[j] < rank[i]:
                max_iou_h = max(max_iou_h, v)
                c = containment(preds[i]["bbox"], preds[j]["bbox"])
                max_cont_h = max(max_cont_h, c)
        results.append((count_nbr, max_iou_h, max_cont_h))
    return results


def cross_detector_features(codetr_by_img, other_preds_list):
    n_other = len(other_preds_list)
    cross = {}
    for img_id, preds in codetr_by_img.items():
        boxes = [p["bbox"] for p in preds]
        for oi, other_by_img in enumerate(other_preds_list):
            other_preds = other_by_img.get(img_id, [])
            other_boxes = [p["bbox"] for p in other_preds]
            other_scores = [p["score"] for p in other_preds]
            if len(other_boxes) == 0:
                for i in range(len(preds)):
                    key = (img_id, i)
                    if key not in cross:
                        cross[key] = [0.0] * (2 * n_other)
                continue
            iou_mat = iou_xyxy_vectorized(boxes, other_boxes)
            for i in range(len(preds)):
                key = (img_id, i)
                if key not in cross:
                    cross[key] = [0.0] * (2 * n_other)
                best_j = int(np.argmax(iou_mat[i]))
                cross[key][2*oi] = float(iou_mat[i, best_j])
                cross[key][2*oi + 1] = (
                    float(other_scores[best_j])
                    if iou_mat[i, best_j] > 0.01 else 0.0)
    return cross


def load_preds_by_img(path):
    with open(path) as f:
        preds = json.load(f)
    by_img = defaultdict(list)
    for p in preds:
        by_img[p["image_id"]].append(p)
    return by_img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--codetr-preds", required=True)
    ap.add_argument("--cascade-preds", required=True)
    ap.add_argument("--dfine-preds", required=True)
    ap.add_argument("--dfine-tta-preds", required=True)
    ap.add_argument("--img-dir", required=True)
    ap.add_argument("--ann", required=True)
    ap.add_argument("--threshold", type=float, default=0.18)
    ap.add_argument("--out", required=True)
    ap.add_argument("--crop-size", type=int, default=96)
    ap.add_argument("--ctx", type=float, default=1.5)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    feat_names = [str(n) for n in ck["feat_names"]]
    feat_mean = np.array(ck["feat_mean"], dtype=np.float32)
    feat_std = np.array(ck["feat_std"], dtype=np.float32)
    n_feat = len(feat_names)

    sd = ck.get("ema_shadow", ck["model"])
    base = sd["stem.conv1.weight"].shape[0]
    model = VerifierV2(n_feat=n_feat, base=base).to(device)
    if "ema_shadow" in ck:
        named_params = dict(model.named_parameters())
        for k, v in sd.items():
            if k in named_params:
                named_params[k].data.copy_(v)
        bn_keys = {k for k in ck["model"]
                   if "running_" in k or "num_batches" in k}
        model_sd = model.state_dict()
        for k in bn_keys:
            if k in model_sd:
                model_sd[k] = ck["model"][k]
        model.load_state_dict(model_sd)
    else:
        model.load_state_dict(sd)
    model.eval()
    print(f"[prune] Loaded enhancer: base={base} n_feat={n_feat} "
          f"epoch={ck.get('epoch','?')} dev_AUC={ck.get('dev_auc',0):.4f}")

    coco = json.load(open(args.ann))
    img_meta = {im["id"]: im for im in coco["images"]}

    print(f"[prune] Loading predictions ...")
    codetr_by_img = load_preds_by_img(args.codetr_preds)
    cascade_by_img = load_preds_by_img(args.cascade_preds)
    dfine_by_img = load_preds_by_img(args.dfine_preds)
    dfine_tta_by_img = load_preds_by_img(args.dfine_tta_preds)

    total_preds = sum(len(v) for v in codetr_by_img.values())
    print(f"  Co-DETR: {total_preds} predictions across "
          f"{len(codetr_by_img)} images")

    print(f"[prune] Computing cross-detector features ...")
    cross = cross_detector_features(
        codetr_by_img, [cascade_by_img, dfine_by_img, dfine_tta_by_img])

    print(f"[prune] Building features and crops ...")
    img_dir = Path(args.img_dir)
    flat = []
    feats_all = []
    crops_all = []
    img_cache = {}

    for img_id in sorted(codetr_by_img.keys()):
        preds = codetr_by_img[img_id]
        if img_id not in img_meta:
            continue
        meta = img_meta[img_id]
        H, W = meta["height"], meta["width"]

        nbrs = neighborhood_features_flat(preds)

        if img_id not in img_cache:
            ip = img_dir / meta["file_name"]
            img_cache[img_id] = np.array(Image.open(ip).convert("RGB"))
        img_arr = img_cache[img_id]

        for i, p in enumerate(preds):
            x, y, w, h = p["bbox"]
            area = max(w * h, 1.0)
            ar = max(w, h) / max(min(w, h), 1.0)
            n_nbr, max_iou_h, max_cont_h = nbrs[i]
            cross_feats = cross.get((img_id, i), [0.0] * 6)

            row = [
                p["score"], np.log(area), np.log(ar),
                (x + w / 2) / W, (y + h / 2) / H,
                w / W, h / H,
                float(n_nbr), max_iou_h, max_cont_h,
                *cross_feats,
            ]
            feats_all.append(row)

            crop = make_crop(img_arr, p["bbox"],
                             crop_size=args.crop_size, ctx=args.ctx)
            crops_all.append(np.transpose(crop, (2, 0, 1)))
            flat.append((img_id, i, p))

        if len(img_cache) > 200:
            for k in list(img_cache.keys())[:100]:
                del img_cache[k]

    feats_arr = np.array(feats_all, dtype=np.float32)
    feats_z = (feats_arr - feat_mean) / (feat_std + 1e-9)
    crops_arr = np.array(crops_all, dtype=np.uint8)

    print(f"[prune] Scoring {len(flat)} predictions "
          f"(batch={args.batch_size}) ...")
    p_tp = np.zeros(len(flat), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(flat), args.batch_size):
            j = min(i + args.batch_size, len(flat))
            c = torch.from_numpy(crops_arr[i:j]).to(device)
            f = torch.from_numpy(feats_z[i:j]).to(device)
            logit = model(c, f)
            p_tp[i:j] = torch.sigmoid(logit).cpu().numpy()

    kept = 0
    out_preds = []
    for idx, (img_id, pred_i, p) in enumerate(flat):
        if p_tp[idx] >= args.threshold:
            out_preds.append(p)
            kept += 1

    pruned = len(flat) - kept
    print(f"[prune] threshold={args.threshold:.2f}: "
          f"kept {kept}/{len(flat)} ({100*kept/max(len(flat),1):.1f}%), "
          f"pruned {pruned} ({100*pruned/max(len(flat),1):.1f}%)")
    print(f"  p_tp stats: mean={p_tp.mean():.4f} median={np.median(p_tp):.4f} "
          f"min={p_tp.min():.4f} max={p_tp.max():.4f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out_preds, f)
    print(f"[prune] Wrote {args.out} ({len(out_preds)} predictions)")


if __name__ == "__main__":
    main()
