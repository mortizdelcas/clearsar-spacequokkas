import argparse
import json
import os
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
DFINE_ROOT = os.environ.get("DFINE_ROOT", str(_HERE.parent.parent / "D-FINE"))
sys.path.insert(0, DFINE_ROOT)
sys.path.insert(0, str(_HERE))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torchvision.transforms as T  # noqa: E402
from PIL import Image  # noqa: E402
from tqdm import tqdm  # noqa: E402

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


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--config", required=True)
    ap.add_argument("--test-dir", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--threshold", type=float, default=0.01)
    ap.add_argument("--input-size", type=int, default=1024)
    ap.add_argument("--apply-clahe", action="store_true")
    ap.add_argument("--device", default="cuda:0")
    return ap.parse_args()


def main():
    args = parse_args()

    test_dir = Path(args.test_dir)
    if not test_dir.exists():
        raise SystemExit(f"--test-dir not found: {test_dir}")
    img_paths = sorted(p for p in test_dir.iterdir()
                       if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
    print(f"Found {len(img_paths)} images in {test_dir}")

    device = torch.device(args.device)
    print(f"Loading model: cfg={args.config}  ckpt={args.checkpoint}")
    model = build_model(args.config, args.checkpoint, device)

    transforms = T.Compose([
        T.Resize((args.input_size, args.input_size)),
        T.ToTensor(),
    ])

    detections = []
    with torch.no_grad():
        for ip in tqdm(img_paths, desc="D-FINE"):
            image_id = int(ip.stem)
            img = Image.open(ip).convert("RGB")
            if args.apply_clahe:
                img = apply_clahe(img)

            w, h = img.size
            orig_size = torch.tensor([[w, h]]).to(device)
            im_t = transforms(img).unsqueeze(0).to(device)

            labels, boxes, scores = model(im_t, orig_size)
            box = boxes[0]
            scr = scores[0]

            mask = scr > args.threshold
            box = box[mask]
            scr = scr[mask]

            for b, s in zip(box, scr):
                x1, y1, x2, y2 = b.tolist()
                detections.append({
                    "image_id": image_id,
                    "category_id": 1,
                    "bbox": [
                        round(x1, 2),
                        round(y1, 2),
                        round(x2 - x1, 2),
                        round(y2 - y1, 2),
                    ],
                    "score": round(s.item(), 5),
                })

    print(f"Total detections (>= score {args.threshold}): {len(detections)}")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(detections, f)
        f.write("\n")
    size_mb = Path(args.output).stat().st_size / 1e6
    print(f"Wrote {args.output} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
