"""Jeu synthétique type BUSI (image + image_mask) pour tester l'entraînement. Pas de données patient."""
import sys
from pathlib import Path
import cv2
import numpy as np


def make(out, n=60, S=160, seed=0):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:S, :S]
    for i in range(n):
        cx, cy = rng.uniform(40, S - 40, 2); a, b = rng.uniform(12, 28, 2); ang = rng.uniform(0, np.pi)
        u = (xx - cx) * np.cos(ang) + (yy - cy) * np.sin(ang); v = -(xx - cx) * np.sin(ang) + (yy - cy) * np.cos(ang)
        m = ((u / a) ** 2 + (v / b) ** 2 < 1).astype(np.uint8)
        img = 110 + 25 * (yy / S) + 0 * xx
        img[m > 0] = 50
        img = cv2.GaussianBlur(img.astype(np.float32), (0, 0), 1.5) * (0.6 + rng.rayleigh(0.3, img.shape))
        cv2.imwrite(str(out / f"benign ({i}).png"), np.clip(img, 0, 255).astype(np.uint8))
        cv2.imwrite(str(out / f"benign ({i})_mask.png"), m * 255)


if __name__ == "__main__":
    make(sys.argv[1])
