import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
_orig_load = torch.load
torch.load = lambda *a, **kw: _orig_load(
    *a, **{**kw, "weights_only": kw.get("weights_only", False)})

from tqdm import tqdm  # noqa: E402
from mmdet.apis import init_detector, inference_detector  # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("checkpoint")
    ap.add_argument("--test-dir", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--score-thr", type=float, default=0.01)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", default="cuda:0")
    return ap.parse_args()


def result_to_dets(img_path, result, score_thr):
    image_id = int(Path(img_path).stem)
    pi = result.pred_instances
    bboxes = pi.bboxes.cpu().numpy()
    scores = pi.scores.cpu().numpy()
    out = []
    for bbox, score in zip(bboxes, scores):
        if score < score_thr:
            continue
        x1, y1, x2, y2 = bbox
        out.append({
            "image_id": image_id,
            "category_id": 1,
            "bbox": [
                round(float(x1), 2),
                round(float(y1), 2),
                round(float(x2 - x1), 2),
                round(float(y2 - y1), 2),
            ],
            "score": round(float(score), 5),
        })
    return out


def main():
    args = parse_args()

    test_dir = Path(args.test_dir)
    if not test_dir.exists():
        raise SystemExit(f"--test-dir not found: {test_dir}")
    img_paths = sorted(p for p in test_dir.iterdir()
                       if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
    print(f"Found {len(img_paths)} images in {test_dir}")

    print(f"Loading model: cfg={args.config}  ckpt={args.checkpoint}")
    model = init_detector(args.config, args.checkpoint, device=args.device)

    detections = []
    for start in tqdm(range(0, len(img_paths), args.batch_size),
                      desc="Inference"):
        batch = [str(p) for p in img_paths[start:start + args.batch_size]]
        results = inference_detector(model, batch)
        for ip, res in zip(batch, results):
            detections.extend(result_to_dets(ip, res, args.score_thr))

    print(f"Total detections (>= score {args.score_thr}): {len(detections)}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(detections, f)
        f.write("\n")
    size_mb = Path(args.output).stat().st_size / 1e6
    print(f"Wrote {args.output} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
