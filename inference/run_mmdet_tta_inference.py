import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import torch  # noqa: E402

_orig_load = torch.load
torch.load = lambda *a, **kw: _orig_load(
    *a, **{**kw, "weights_only": kw.get("weights_only", False)})

from PIL import Image  # noqa: E402
from tqdm import tqdm  # noqa: E402
from ensemble_boxes import weighted_boxes_fusion  # noqa: E402
from mmdet.apis import init_detector, inference_detector  # noqa: E402


def aug_image(img_np, aug):
    if aug == "base":
        return img_np
    if aug == "hflip":
        return img_np[:, ::-1, :].copy()
    if aug == "vflip":
        return img_np[::-1, :, :].copy()
    if aug == "hvflip":
        return img_np[::-1, ::-1, :].copy()
    raise ValueError(f"unknown aug: {aug}")


def unaug_boxes(boxes, aug, w, h):
    if len(boxes) == 0 or aug == "base":
        return boxes
    out = boxes.copy()
    if aug in ("hflip", "hvflip"):
        new_x1 = w - out[:, 2]
        new_x2 = w - out[:, 0]
        out[:, 0] = new_x1
        out[:, 2] = new_x2
    if aug in ("vflip", "hvflip"):
        new_y1 = h - out[:, 3]
        new_y2 = h - out[:, 1]
        out[:, 1] = new_y1
        out[:, 3] = new_y2
    return out


def merge_with_wbf(boxes_per_aug, scores_per_aug, w, h, weights, iou_thr,
                   skip_box_thr=0.0001, conf_type="max"):
    bl, sl, ll = [], [], []
    for boxes, scores in zip(boxes_per_aug, scores_per_aug):
        if len(boxes) == 0:
            bl.append(np.zeros((0, 4)))
            sl.append(np.array([]))
            ll.append(np.array([]))
            continue
        bb = boxes.copy()
        bb[:, [0, 2]] /= w
        bb[:, [1, 3]] /= h
        bl.append(np.clip(bb, 0, 1))
        sl.append(scores)
        ll.append(np.zeros(len(scores)))
    fb, fs, _ = weighted_boxes_fusion(
        bl, sl, ll, weights=weights, iou_thr=iou_thr,
        skip_box_thr=skip_box_thr, conf_type=conf_type)
    if len(fb) > 0:
        fb[:, [0, 2]] *= w
        fb[:, [1, 3]] *= h
    return fb, fs


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("checkpoint")
    ap.add_argument("--test-dir", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--augs", default="base,hflip",
                    help="Comma-separated subset of base,hflip,vflip,hvflip.")
    ap.add_argument("--iou-thr", type=float, default=0.7)
    ap.add_argument("--score-thr", type=float, default=0.001)
    ap.add_argument("--final-thr", type=float, default=0.001)
    ap.add_argument("--conf-type", default="max",
                    choices=["avg", "max", "box_and_model_avg",
                             "absent_model_aware_avg"])
    ap.add_argument("--weights", default=None)
    ap.add_argument("--device", default="cuda:0")
    return ap.parse_args()


def main():
    args = parse_args()
    augs = [a.strip() for a in args.augs.split(",") if a.strip()]
    weights = ([float(w) for w in args.weights.split(",")] if args.weights
               else [1.0] * len(augs))
    assert len(weights) == len(augs)
    print(f"Augs: {augs}  weights: {weights}  iou={args.iou_thr} "
          f"conf_type={args.conf_type}")

    test_dir = Path(args.test_dir)
    if not test_dir.exists():
        raise SystemExit(f"--test-dir not found: {test_dir}")
    img_paths = sorted(p for p in test_dir.iterdir()
                       if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
    print(f"Found {len(img_paths)} images in {test_dir}")

    print(f"Loading model from {args.checkpoint} ...")
    model = init_detector(args.config, args.checkpoint, device=args.device)

    out_dets = []
    for ip in tqdm(img_paths, desc="TTA"):
        image_id = int(ip.stem)
        img_np = np.array(Image.open(ip).convert("RGB"))
        h, w = img_np.shape[:2]

        boxes_per, scores_per = [], []
        for aug in augs:
            res = inference_detector(model, aug_image(img_np, aug))
            pi = res.pred_instances
            boxes = pi.bboxes.cpu().numpy()
            scores = pi.scores.cpu().numpy()
            m = scores >= args.score_thr
            boxes, scores = boxes[m], scores[m]
            boxes = unaug_boxes(boxes, aug, w, h)
            boxes_per.append(boxes)
            scores_per.append(scores)

        if len(augs) == 1:
            fb, fs = boxes_per[0], scores_per[0]
        else:
            fb, fs = merge_with_wbf(boxes_per, scores_per, w, h, weights,
                                    args.iou_thr, conf_type=args.conf_type)
        keep = fs >= args.final_thr
        for box, s in zip(fb[keep], fs[keep]):
            x1, y1, x2, y2 = box.tolist()
            out_dets.append({
                "image_id": image_id, "category_id": 1,
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "score": float(s),
            })

    print(f"Total fused detections: {len(out_dets)}")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out_dets, f)
    size_mb = Path(args.output).stat().st_size / 1e6
    print(f"Wrote {args.output} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
