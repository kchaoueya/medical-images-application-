"""Lecture / encodage d'images."""
import base64
import cv2
import numpy as np

MAX_SIDE = 1024


def decode_gray(data: bytes) -> np.ndarray:
    """Décode un fichier image (png/jpg/bmp/tif, 8 ou 16 bits) en niveaux de gris uint8."""
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("Fichier image illisible (formats : PNG, JPG, BMP, TIFF).")
    if img.ndim == 3:
        if img.shape[2] == 4:
            img = img[:, :, :3]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.dtype != np.uint8:  # 16 bits, float...
        img = img.astype(np.float64)
        lo, hi = img.min(), img.max()
        img = ((img - lo) / (hi - lo + 1e-9) * 255).astype(np.uint8)
    h, w = img.shape
    s = MAX_SIDE / max(h, w)
    if s < 1:
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    return img


def decode_mask(data: bytes, shape) -> np.ndarray:
    """Décode un masque de référence, le met à la taille de l'image (plus proche voisin) et binarise."""
    arr = np.frombuffer(data, np.uint8)
    m = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise ValueError("Masque illisible.")
    m = cv2.resize(m, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return (m > 127).astype(np.uint8)


def to_data_url(img: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", img)
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode()


def png_bytes(img: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", img)
    return buf.tobytes()


PALETTE = [  # BGR, lisibles sur fond gris
    (60, 76, 231), (219, 152, 52), (113, 204, 46), (15, 196, 241),
    (182, 89, 155), (156, 188, 26), (34, 126, 230), (166, 166, 127),
]
PALETTE_HEX = ["#%02x%02x%02x" % (c[2], c[1], c[0]) for c in PALETTE]


def overlay(gray: np.ndarray, mask: np.ndarray, color=(60, 76, 231), alpha=0.4) -> np.ndarray:
    """Superpose un masque (remplissage translucide + contour) sur l'image."""
    bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    layer = bgr.copy()
    layer[mask > 0] = color
    out = cv2.addWeighted(layer, alpha, bgr, 1 - alpha, 0)
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(out, cnts, -1, color, 2)
    return out


def contours_overlay(gray, masks_colors, gt=None):
    """Contours de plusieurs masques (liste de (mask, bgr)) sur l'image ; référence en jaune pointillé épais."""
    out = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    if gt is not None:
        c, _ = cv2.findContours(gt.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(out, c, -1, (0, 255, 255), 3)
    for m, col in masks_colors:
        c, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(out, c, -1, col, 2)
    return out
