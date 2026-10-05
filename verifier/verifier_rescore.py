import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from verifier_features import neighborhood_features, make_crop
from verifier_model import VerifierV2, Verifier


PROVENANCE_LABELS_4 = ("A", "B", "C", "CT")
PROVENANCE_LABELS_6 = ("A", "B", "C", "CT", "E", "ET")


def build_features(preds_by_img, img_meta, feat_names_expected, n_src):
    nbrs = neighborhood_features(preds_by_img)

    if n_src == 4:
        src_labels = PROVENANCE_LABELS_4
    elif n_src == 6:
        src_labels = PROVENANCE_LABELS_6
    else:
        src_labels = [f"S{i}" for i in range(n_src)]

    base_names = [
        "score", "area_log", "ar_log", "cx_norm", "cy_norm",
        "w_norm", "h_norm", "n_neighbors", "max_iou_higher", "max_cont_higher",
    ]
    full_names = base_names.copy()
    full_names += [f"src_{s}_pres" for s in src_labels]
    full_names += [f"src_{s}_score" for s in src_labels]
    full_names += [f"src_{s}_iou" for s in src_labels]

    flat, feats = [], []
    for img_id, pp in preds_by_img.items():
        if img_id not in img_meta:
            continue
        meta = img_meta[img_id]
        H, W = meta["height"], meta["width"]
        for p in pp:
            x, y, w, h = p["bbox"]
            area = max(w * h, 1.0)
            ar = max(w, h) / max(min(w, h), 1.0)
            n_nbr, max_iou_h, max_cont_h = nbrs[(img_id, id(p))]
            row = [
                p["score"], np.log(area), np.log(ar),
                (x + w / 2) / W, (y + h / 2) / H,
                w / W, h / H,
                n_nbr, max_iou_h, max_cont_h,
                *p["src_present"], *p["src_score"], *p["src_max_iou"],
            ]
            feats.append(row)
            flat.append((img_id, p))

    feats_arr = np.array(feats, dtype=np.float32)

    if len(full_names) == len(feat_names_expected):
        return flat, feats_arr, full_names

    name_to_col = {n: i for i, n in enumerate(full_names)}
    out_feats = np.zeros((len(feats), len(feat_names_expected)),
                         dtype=np.float32)
    for j, name in enumerate(feat_names_expected):
        if name in name_to_col:
            out_feats[:, j] = feats_arr[:, name_to_col[name]]
    print(f"  Feature mapping: {len(full_names)} -> "
          f"{len(feat_names_expected)} "
          f"(padded {sum(1 for n in feat_names_expected if n not in name_to_col)} zeros)")
    return flat, out_feats, feat_names_expected


def load_model_from_ckpt(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    feat_mean = np.array(ck["feat_mean"], dtype=np.float32)
    feat_std = np.array(ck["feat_std"], dtype=np.float32)
    feat_names = [str(n) for n in ck["feat_names"]]

    sd = ck.get("ema_shadow", ck["model"])
    if isinstance(sd, dict) and any(
            k.startswith("stem.se.") or k.startswith("b1.se.")
            for k in sd.keys()):
        model_type = "v2"
    else:
        model_type = "legacy"

    n_feat = len(feat_names)
    base = sd["stem.conv1.weight"].shape[0]
    if model_type == "v2":
        model = VerifierV2(n_feat=n_feat, base=base).to(device)
    else:
        model = Verifier(n_feat=n_feat, base=base).to(device)

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
    print(f"  Loaded {ckpt_path}: {model_type} base={base} "
          f"n_feat={n_feat} epoch={ck.get('epoch', '?')} "
          f"dev_AUC={ck.get('dev_auc', 0):.4f}")
    return model, feat_mean, feat_std, feat_names


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True,
                    help="Comma-separated checkpoints (ensemble).")
    ap.add_argument("--fused-preds", required=True,
                    help="Fused-WBF predictions WITH provenance.")
    ap.add_argument("--ann", required=True)
    ap.add_argument("--img-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["mult", "blend_band"],
                    default="blend_band")
    ap.add_argument("--alpha", type=float, default=0.5,
                    help="mode=mult: score' = score * p_tp**alpha")
    ap.add_argument("--cap", type=float, default=0.87,
                    help="mode=blend_band: only blend boxes with "
                         "raw score < cap.")
    ap.add_argument("--w", type=float, default=0.85,
                    help="mode=blend_band: convex-blend weight.")
    ap.add_argument("--score-min", type=float, default=0.0)
    ap.add_argument("--crop-size", type=int, default=96)
    ap.add_argument("--ctx", type=float, default=1.5)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--device", default="cuda:0")
    return ap.parse_args()


def main():
    args = parse_args()
    ckpt_paths = [p.strip() for p in args.ckpt.split(",")]
    n_models = len(ckpt_paths)
    print(f"[rescore] Loading {n_models} checkpoint(s)")

    models, feat_means, feat_stds, feat_names_list = [], [], [], []
    for p in ckpt_paths:
        m, fm, fs, fn = load_model_from_ckpt(p, args.device)
        models.append(m)
        feat_means.append(fm)
        feat_stds.append(fs)
        feat_names_list.append(fn)

    print(f"\n[rescore] loading {args.fused_preds}")
    preds = json.load(open(args.fused_preds))
    print(f"  {len(preds)} predictions")
    if "src_present" not in preds[0]:
        raise SystemExit("ERROR: input has no provenance fields.")
    n_src = len(preds[0]["src_present"])
    print(f"  Provenance sources: {n_src}")
    if args.score_min > 0:
        preds = [d for d in preds if d["score"] >= args.score_min]
        print(f"  after score>={args.score_min}: {len(preds)} predictions")

    print(f"[rescore] loading image meta from {args.ann}")
    coco = json.load(open(args.ann))
    img_meta = {im["id"]: im for im in coco["images"]}

    preds_by_img = defaultdict(list)
    for p in preds:
        preds_by_img[p["image_id"]].append(p)

    all_feats_z = []
    flat = None
    for i, fn in enumerate(feat_names_list):
        f, feats_arr, _ = build_features(preds_by_img, img_meta, fn, n_src)
        if flat is None:
            flat = f
        fz = (feats_arr - feat_means[i]) / (feat_stds[i] + 1e-9)
        all_feats_z.append(fz)

    print("\n[rescore] extracting crops ...")
    img_dir = Path(args.img_dir)
    img_cache = {}
    crops = np.zeros((len(flat), 3, args.crop_size, args.crop_size),
                     dtype=np.uint8)
    for i, (img_id, p) in enumerate(tqdm(flat)):
        meta = img_meta[img_id]
        if img_id not in img_cache:
            ip = img_dir / meta["file_name"]
            img_cache[img_id] = np.array(Image.open(ip).convert("RGB"))
        crop = make_crop(img_cache[img_id], p["bbox"],
                         crop_size=args.crop_size, ctx=args.ctx)
        crops[i] = np.transpose(crop, (2, 0, 1))
        if len(img_cache) > 200:
            for k in list(img_cache.keys())[:100]:
                del img_cache[k]

    print(f"[rescore] scoring {len(crops)} preds "
          f"(batch={args.batch_size}, models={n_models}) ...")
    p_tp_all = np.zeros((n_models, len(crops)), dtype=np.float32)

    for mi, (model, feats_z) in enumerate(zip(models, all_feats_z)):
        with torch.no_grad():
            for i in range(0, len(crops), args.batch_size):
                j = min(i + args.batch_size, len(crops))
                c = torch.from_numpy(crops[i:j]).to(args.device)
                f = torch.from_numpy(feats_z[i:j]).to(args.device)
                logit = model(c, f)
                p_tp_all[mi, i:j] = torch.sigmoid(logit).cpu().numpy()

    p_tp = p_tp_all.mean(axis=0)

    raw_scores = np.array([p["score"] for _, p in flat], dtype=np.float64)
    if args.mode == "mult":
        p_tp_safe = np.clip(p_tp, 1e-6, 1.0)
        new_scores = raw_scores * (p_tp_safe ** float(args.alpha))
    elif args.mode == "blend_band":
        new_scores = raw_scores.copy()
        mask = raw_scores < float(args.cap)
        w = float(args.w)
        new_scores[mask] = ((1 - w) * raw_scores[mask]
                            + w * p_tp[mask].astype(np.float64))
    else:
        raise ValueError(f"unknown mode: {args.mode}")

    out_preds = []
    for (img_id, p), new_s in zip(flat, new_scores):
        out_preds.append({
            "image_id": int(p["image_id"]),
            "category_id": int(p["category_id"]),
            "bbox": [float(b) for b in p["bbox"]],
            "score": float(new_s),
        })

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out_preds, f)

    print(f"\n[rescore] wrote {args.out}  ({len(out_preds)} preds)")
    if args.mode == "mult":
        print(f"  mode=mult  alpha={args.alpha}")
    else:
        n_blend = int((raw_scores < args.cap).sum())
        print(f"  mode=blend_band  cap={args.cap}  w={args.w}  "
              f"({n_blend}/{len(raw_scores)} boxes blended)")
    print(f"  raw  score: mean={raw_scores.mean():.4f}  "
          f"median={np.median(raw_scores):.4f}")
    print(f"  p_tp:       mean={p_tp.mean():.4f}  "
          f"median={np.median(p_tp):.4f}  "
          f">0.5: {(p_tp > 0.5).sum()}/{len(p_tp)}")
    print(f"  new  score: mean={new_scores.mean():.4f}  "
          f"median={np.median(new_scores):.4f}  "
          f"min={new_scores.min():.6f}  max={new_scores.max():.4f}")
    if n_models > 1:
        stds = p_tp_all.std(axis=0)
        print(f"  ensemble std: mean={stds.mean():.4f}  "
              f"max={stds.max():.4f}")


if __name__ == "__main__":
    main()
