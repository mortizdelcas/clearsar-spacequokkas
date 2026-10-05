import argparse
import json
import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
DFINE_ROOT = os.environ.get("DFINE_ROOT", str(_HERE.parent.parent / "D-FINE"))
sys.path.insert(0, DFINE_ROOT)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torchvision.transforms.functional as TF  # noqa: E402
from PIL import Image  # noqa: E402
from tqdm import tqdm  # noqa: E402
from ensemble_boxes import weighted_boxes_fusion  # noqa: E402

from src.core import YAMLConfig  # noqa: E402


def apply_clahe(img_pil, clip_limit=4.0, tile_grid_size=(8, 8)):
    import cv2
    img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(img_cv, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    img_cv = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))


def build_model(config_path, checkpoint_path, device):
    cfg = YAMLConfig(config_path, resume=checkpoint_path)
    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = ckpt["ema"]["module"] if "ema" in ckpt else ckpt["model"]
    cfg.model.load_state_dict(state)

    class DeployModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()

        def forward(self, images, orig_target_sizes):
            return self.postprocessor(self.model(images), orig_target_sizes)

    return DeployModel().to(device).eval()


def parse_augs(s):
    """Each aug is 'SIZE' or 'SIZE_hflip' (e.g. '1024,1280,1024_hflip')."""
    out = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        hflip = False
        if part.endswith("_hflip"):
            hflip = True
            part = part[:-len("_hflip")]
        out.append((int(part), hflip))
    return out


def run_one(model, img, size, hflip, device):
    w, h = img.size
    img_in = TF.hflip(img) if hflip else img
    img_t = TF.resize(img_in, [size, size])
    img_t = TF.to_tensor(img_t).unsqueeze(0).to(device)
    orig_size = torch.tensor([[w, h]]).to(device)
    with torch.no_grad():
        labels, boxes, scores = model(img_t, orig_size)
    box = boxes[0].cpu().numpy()
    scr = scores[0].cpu().numpy()
    if hflip and len(box) > 0:
        new_x1 = w - box[:, 2]
        new_x2 = w - box[:, 0]
        box[:, 0] = new_x1
        box[:, 2] = new_x2
    return box, scr


def merge_with_wbf(boxes_per_aug, scores_per_aug, w, h, weights, iou_thr,
                   score_thr=0.001, skip_box_thr=0.0001,
                   conf_type="absent_model_aware_avg"):
    bl, sl, ll = [], [], []
    for boxes, scores in zip(boxes_per_aug, scores_per_aug):
        if len(boxes) == 0:
            bl.append(np.zeros((0, 4)))
            sl.append(np.array([]))
            ll.append(np.array([]))
            continue
        m = scores >= score_thr
        if not np.any(m):
            bl.append(np.zeros((0, 4)))
            sl.append(np.array([]))
            ll.append(np.array([]))
            continue
        bb = boxes[m].copy()
        bb[:, [0, 2]] /= w
        bb[:, [1, 3]] /= h
        bl.append(np.clip(bb, 0, 1))
        sl.append(scores[m])
        ll.append(np.zeros(int(m.sum())))
    fb, fs, _ = weighted_boxes_fusion(
        bl, sl, ll, weights=weights, iou_thr=iou_thr,
        skip_box_thr=skip_box_thr, conf_type=conf_type)
    if len(fb) > 0:
        fb[:, [0, 2]] *= w
        fb[:, [1, 3]] *= h
    return fb, fs


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--config", required=True)
    ap.add_argument("--test-dir", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--apply-clahe", action="store_true")
    ap.add_argument("--augs", default="1024,1024_hflip")
    ap.add_argument("--weights", default=None)
    ap.add_argument("--iou-thr", type=float, default=0.7)
    ap.add_argument("--score-thr", type=float, default=0.001)
    ap.add_argument("--final-thr", type=float, default=0.001)
    ap.add_argument("--conf-type", default="max",
                    choices=["avg", "max", "box_and_model_avg",
                             "absent_model_aware_avg"])
    ap.add_argument("--device", default="cuda:0")
    return ap.parse_args()


def main():
    args = parse_args()

    augs = parse_augs(args.augs)
    print(f"TTA augmentations ({len(augs)}):")
    for s, hf in augs:
        print(f"  size={s} hflip={hf}")
    weights = ([float(w) for w in args.weights.split(",")] if args.weights
               else [1.0] * len(augs))
    assert len(weights) == len(augs), "len(weights) must match len(augs)"
    print(f"Per-aug weights: {weights}, IoU={args.iou_thr}, "
          f"conf_type={args.conf_type}, apply_clahe={args.apply_clahe}")

    test_dir = Path(args.test_dir)
    if not test_dir.exists():
        raise SystemExit(f"--test-dir not found: {test_dir}")
    img_paths = sorted(p for p in test_dir.iterdir()
                       if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
    print(f"Found {len(img_paths)} images in {test_dir}")

    device = torch.device(args.device)
    print(f"Loading model: cfg={args.config}  ckpt={args.checkpoint}")
    model = build_model(args.config, args.checkpoint, device)

    detections = []
    for ip in tqdm(img_paths, desc="D-FINE TTA"):
        image_id = int(ip.stem)
        img = Image.open(ip).convert("RGB")
        if args.apply_clahe:
            img = apply_clahe(img)
        w, h = img.size

        boxes_per, scores_per = [], []
        for size, hflip in augs:
            b, s = run_one(model, img, size, hflip, device)
            boxes_per.append(b)
            scores_per.append(s)

        fb, fs = merge_with_wbf(boxes_per, scores_per, w, h, weights,
                                args.iou_thr, score_thr=args.score_thr,
                                conf_type=args.conf_type)
        keep = fs >= args.final_thr
        for box, score in zip(fb[keep], fs[keep]):
            x1, y1, x2, y2 = box.tolist()
            detections.append({
                "image_id": image_id, "category_id": 1,
                "bbox": [round(x1, 2), round(y1, 2),
                         round(x2 - x1, 2), round(y2 - y1, 2)],
                "score": round(float(score), 5),
            })

    print(f"Total fused detections: {len(detections)}")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(detections, f)
        f.write("\n")
    size_mb = Path(args.output).stat().st_size / 1e6
    print(f"Wrote {args.output} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
