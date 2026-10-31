"""Mesures : Dice, IoU, descripteurs de forme, distance de Hausdorff."""
import cv2
import numpy as np
from scipy.spatial.distance import directed_hausdorff


def dice(a, b, eps=1e-7):
    a, b = a.astype(bool), b.astype(bool)
    return float((2 * (a & b).sum() + eps) / (a.sum() + b.sum() + eps))


def iou(a, b, eps=1e-7):
    a, b = a.astype(bool), b.astype(bool)
    return float(((a & b).sum() + eps) / ((a | b).sum() + eps))


def hausdorff(a, b):
    ca, _ = cv2.findContours(a.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cb, _ = cv2.findContours(b.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not ca or not cb:
        return None
    pa = np.vstack([c.reshape(-1, 2) for c in ca])
    pb = np.vstack([c.reshape(-1, 2) for c in cb])
    return float(max(directed_hausdorff(pa, pb)[0], directed_hausdorff(pb, pa)[0]))


def shape_stats(mask, gray):
    m = mask.astype(np.uint8)
    area = int(m.sum())
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    perim = float(sum(cv2.arcLength(c, True) for c in cnts))
    inside = gray[m > 0]
    outside = gray[m == 0]
    return {
        "area_px": area,
        "area_pct": round(100.0 * area / m.size, 2),
        "perimeter": round(perim, 1),
        "circularity": round(float(4 * np.pi * area / perim ** 2), 3) if perim > 0 else 0.0,
        "components": len(cnts),
        "mean_in": round(float(inside.mean()), 1) if inside.size else None,
        "mean_out": round(float(outside.mean()), 1) if outside.size else None,
    }


def pairwise(masks: dict):
    names = list(masks)
    d = [[dice(masks[a], masks[b]) for b in names] for a in names]
    i = [[iou(masks[a], masks[b]) for b in names] for a in names]
    return {"names": names, "dice": d, "iou": i}
