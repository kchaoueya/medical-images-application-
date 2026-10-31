"""Génère des fantômes synthétiques (image + masque de référence) pour tester l'application.
Ce ne sont PAS des images de patients."""
import sys
from pathlib import Path
import cv2
import numpy as np

out = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent / "samples")
out.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(1)
S = 384
yy, xx = np.mgrid[:S, :S]


def blob(cx, cy, a, b, ang, wob=0.08, seed=0):
    r = np.random.default_rng(seed)
    t = np.arctan2(yy - cy, xx - cx) - ang
    d = np.hypot((xx - cx) * np.cos(ang) + (yy - cy) * np.sin(ang), -(xx - cx) * np.sin(ang) + (yy - cy) * np.cos(ang))
    th = np.arctan2(-(xx - cx) * np.sin(ang) + (yy - cy) * np.cos(ang), (xx - cx) * np.cos(ang) + (yy - cy) * np.sin(ang))
    rad = (a * b) / np.sqrt((b * np.cos(th)) ** 2 + (a * np.sin(th)) ** 2)
    rad = rad * (1 + wob * np.sin(3 * th + r.uniform(0, 6)) + wob / 2 * np.sin(5 * th + r.uniform(0, 6)))
    return (d < rad).astype(np.uint8)


# 1) Échographie type sein : lésion hypoéchogène (sombre) + speckle
bg = 110 + 25 * np.sin(yy / 40.0) + 20 * (yy / S)
les = blob(200, 190, 62, 42, 0.4, seed=1)
img = bg.copy()
img[les > 0] = 45
img = cv2.GaussianBlur(img, (0, 0), 2)
speckle = rng.rayleigh(0.35, img.shape)
img = np.clip(img * (0.55 + speckle), 0, 255).astype(np.uint8)
img = cv2.GaussianBlur(img, (0, 0), 1.2)
cv2.imwrite(str(out / "echo_lesion.png"), img)
cv2.imwrite(str(out / "echo_lesion_mask.png"), les * 255)

# 2) IRM cérébrale : disque clair sur fond noir, tumeur plus claire
head = blob(192, 192, 150, 165, 0, wob=0.03, seed=2)
brain = blob(192, 192, 125, 140, 0, wob=0.04, seed=3)
tum = blob(235, 150, 36, 30, 0.7, wob=0.1, seed=4)
img = np.zeros((S, S), np.float32)
img[head > 0] = 90
img[brain > 0] = 120 + 15 * np.sin(xx[brain > 0] / 15.0)
vent = blob(185, 200, 22, 12, 0.2, seed=5)
img[vent > 0] = 50
img[tum > 0] = 215
img = cv2.GaussianBlur(img, (0, 0), 1.5) + rng.normal(0, 6, img.shape)
cv2.imwrite(str(out / "irm_tumeur.png"), np.clip(img, 0, 255).astype(np.uint8))
cv2.imwrite(str(out / "irm_tumeur_mask.png"), tum * 255)

# 3) Endoscopie type cordes vocales : champ circulaire rosé, glotte sombre
field = blob(192, 192, 175, 175, 0, wob=0.01, seed=6)
gl = blob(192, 200, 22, 70, 1.57, wob=0.07, seed=7)
img = np.zeros((S, S), np.float32)
img[field > 0] = (150 - 40 * np.hypot(xx - 192, yy - 192) / 175)[field > 0]
img[gl > 0] = 25
img = cv2.GaussianBlur(img, (0, 0), 2)
for _ in range(6):  # reflets
    cv2.circle(img, (int(rng.integers(90, 300)), int(rng.integers(90, 300))), int(rng.integers(3, 7)), 250, -1)
img = img + rng.normal(0, 5, img.shape)
cv2.imwrite(str(out / "endoscopie_glotte.png"), np.clip(img, 0, 255).astype(np.uint8))
cv2.imwrite(str(out / "endoscopie_glotte_mask.png"), gl * 255)
print("ok", sorted(p.name for p in out.iterdir()))
