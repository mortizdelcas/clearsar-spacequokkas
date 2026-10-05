import numpy as np


def iou_xywh(b1, b2):
    x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
    x2 = min(b1[0] + b1[2], b2[0] + b2[2])
    y2 = min(b1[1] + b1[3], b2[1] + b2[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    union = b1[2] * b1[3] + b2[2] * b2[3] - inter
    return inter / union if union > 0 else 0.0


def containment(small, big):
    x1 = max(small[0], big[0]); y1 = max(small[1], big[1])
    x2 = min(small[0] + small[2], big[0] + big[2])
    y2 = min(small[1] + small[3], big[1] + big[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    return ((x2 - x1) * (y2 - y1)) / max(small[2] * small[3], 1.0)


def neighborhood_features(preds_by_img):
    nbrs = {}
    for img_id, preds in preds_by_img.items():
        order = sorted(range(len(preds)), key=lambda i: -preds[i]["score"])
        rank = {idx: r for r, idx in enumerate(order)}
        for i in range(len(preds)):
            count_nbr = 0
            max_iou_higher = 0.0
            max_cont_higher = 0.0
            for j in range(len(preds)):
                if i == j:
                    continue
                v = iou_xywh(preds[i]["bbox"], preds[j]["bbox"])
                if v >= 0.5:
                    count_nbr += 1
                if rank[j] < rank[i]:
                    if v > max_iou_higher:
                        max_iou_higher = v
                    c = containment(preds[i]["bbox"], preds[j]["bbox"])
                    if c > max_cont_higher:
                        max_cont_higher = c
            nbrs[(img_id, id(preds[i]))] = (count_nbr, max_iou_higher,
                                             max_cont_higher)
    return nbrs


def make_crop(img_array, bbox, crop_size=96, ctx=1.5):
    x, y, w, h = bbox
    cx = x + w / 2; cy = y + h / 2
    side = max(w, h, 16) * ctx
    H, W = img_array.shape[:2]
    half = side / 2
    x0 = int(round(cx - half)); y0 = int(round(cy - half))
    x1 = int(round(cx + half)); y1 = int(round(cy + half))
    pad_l = max(0, -x0); pad_t = max(0, -y0)
    pad_r = max(0, x1 - W); pad_b = max(0, y1 - H)
    x0 = max(0, x0); y0 = max(0, y0)
    x1 = min(W, x1); y1 = min(H, y1)
    sub = img_array[y0:y1, x0:x1]
    if pad_l or pad_t or pad_r or pad_b:
        sub = np.pad(sub, ((pad_t, pad_b), (pad_l, pad_r), (0, 0)),
                     mode='edge')
    if sub.shape[0] != crop_size or sub.shape[1] != crop_size:
        from PIL import Image as I
        sub = np.array(I.fromarray(sub).resize((crop_size, crop_size),
                                               I.BILINEAR))
    return sub.astype(np.uint8)
