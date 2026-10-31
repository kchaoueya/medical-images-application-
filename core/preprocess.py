"""Modules de prétraitement : filtrage, rehaussement de contraste, réduction du bruit."""
import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def _odd(v, lo=1):
    v = max(int(v), lo)
    return v if v % 2 == 1 else v + 1


def _gaussian(g, p):
    k = _odd(p["ksize"])
    return cv2.GaussianBlur(g, (k, k), p.get("sigma", 0))


def _median(g, p):
    return cv2.medianBlur(g, _odd(p["ksize"], 3))


def _mean(g, p):
    k = _odd(p["ksize"])
    return cv2.blur(g, (k, k))


def _bilateral(g, p):
    return cv2.bilateralFilter(g, int(p["d"]), float(p["sigma_color"]), float(p["sigma_space"]))


def _nlm(g, p):
    return cv2.fastNlMeansDenoising(g, None, h=float(p["h"]), templateWindowSize=7, searchWindowSize=21)


def _clahe(g, p):
    return cv2.createCLAHE(clipLimit=float(p["clip"]), tileGridSize=(int(p["tile"]),) * 2).apply(g)


def _hist_eq(g, p):
    return cv2.equalizeHist(g)


def _gamma(g, p):
    lut = (255 * (np.arange(256) / 255.0) ** (1.0 / float(p["gamma"]))).astype(np.uint8)
    return cv2.LUT(g, lut)


def _stretch(g, p):
    lo, hi = np.percentile(g, [float(p["low"]), float(p["high"])])
    if hi - lo < 1:
        return g
    return np.clip((g.astype(np.float32) - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)


def _unsharp(g, p):
    blur = cv2.GaussianBlur(g, (0, 0), float(p["sigma"]))
    return cv2.addWeighted(g, 1 + float(p["amount"]), blur, -float(p["amount"]), 0)


def _noise(g, p):
    rng = np.random.default_rng(0)
    if p["type"] == "gaussien":
        return np.clip(g + rng.normal(0, float(p["level"]), g.shape), 0, 255).astype(np.uint8)
    out = g.copy()
    r = rng.random(g.shape)
    amt = float(p["level"]) / 100.0
    out[r < amt / 2] = 0
    out[r > 1 - amt / 2] = 255
    return out


def _morph(g, p):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_odd(p["ksize"]),) * 2)
    op = {"ouverture": cv2.MORPH_OPEN, "fermeture": cv2.MORPH_CLOSE,
          "gradient": cv2.MORPH_GRADIENT, "top-hat": cv2.MORPH_TOPHAT, "black-hat": cv2.MORPH_BLACKHAT}[p["op"]]
    return cv2.morphologyEx(g, op, k)


def _invert(g, p):
    return 255 - g


OPS = {
    "gaussian": _gaussian, "median": _median, "mean": _mean, "bilateral": _bilateral, "nlm": _nlm,
    "clahe": _clahe, "hist_eq": _hist_eq, "gamma": _gamma, "stretch": _stretch, "unsharp": _unsharp,
    "noise": _noise, "morph": _morph, "invert": _invert,
}

SCHEMA = [
    {"id": "noise", "name": "Ajouter du bruit (simulation)", "group": "Bruit", "params": [
        {"key": "type", "label": "Type", "type": "select", "options": ["gaussien", "poivre et sel"], "default": "gaussien"},
        {"key": "level", "label": "Niveau (σ ou %)", "type": "float", "min": 1, "max": 60, "step": 1, "default": 25}]},
    {"id": "median", "name": "Filtre médian", "group": "Réduction du bruit", "params": [
        {"key": "ksize", "label": "Noyau", "type": "int", "min": 3, "max": 15, "step": 2, "default": 5}]},
    {"id": "gaussian", "name": "Filtre gaussien", "group": "Réduction du bruit", "params": [
        {"key": "ksize", "label": "Noyau", "type": "int", "min": 3, "max": 21, "step": 2, "default": 5}]},
    {"id": "mean", "name": "Filtre moyenneur", "group": "Réduction du bruit", "params": [
        {"key": "ksize", "label": "Noyau", "type": "int", "min": 3, "max": 21, "step": 2, "default": 5}]},
    {"id": "bilateral", "name": "Filtre bilatéral", "group": "Réduction du bruit", "params": [
        {"key": "d", "label": "Diamètre", "type": "int", "min": 3, "max": 15, "step": 2, "default": 9},
        {"key": "sigma_color", "label": "σ couleur", "type": "float", "min": 10, "max": 200, "step": 5, "default": 75},
        {"key": "sigma_space", "label": "σ espace", "type": "float", "min": 10, "max": 200, "step": 5, "default": 75}]},
    {"id": "nlm", "name": "Non-local means", "group": "Réduction du bruit", "params": [
        {"key": "h", "label": "Force h", "type": "float", "min": 3, "max": 30, "step": 1, "default": 10}]},
    {"id": "clahe", "name": "CLAHE", "group": "Contraste", "params": [
        {"key": "clip", "label": "Clip limit", "type": "float", "min": 1, "max": 10, "step": 0.5, "default": 2.0},
        {"key": "tile", "label": "Grille", "type": "int", "min": 2, "max": 16, "step": 1, "default": 8}]},
    {"id": "hist_eq", "name": "Égalisation d'histogramme", "group": "Contraste", "params": []},
    {"id": "stretch", "name": "Étirement de contraste", "group": "Contraste", "params": [
        {"key": "low", "label": "Percentile bas", "type": "float", "min": 0, "max": 20, "step": 0.5, "default": 1},
        {"key": "high", "label": "Percentile haut", "type": "float", "min": 80, "max": 100, "step": 0.5, "default": 99}]},
    {"id": "gamma", "name": "Correction gamma", "group": "Contraste", "params": [
        {"key": "gamma", "label": "γ", "type": "float", "min": 0.3, "max": 3, "step": 0.1, "default": 1.2}]},
    {"id": "unsharp", "name": "Rehaussement des contours (unsharp)", "group": "Contraste", "params": [
        {"key": "sigma", "label": "σ", "type": "float", "min": 0.5, "max": 5, "step": 0.5, "default": 1.5},
        {"key": "amount", "label": "Intensité", "type": "float", "min": 0.2, "max": 3, "step": 0.1, "default": 1.0}]},
    {"id": "morph", "name": "Morphologie", "group": "Autres", "params": [
        {"key": "op", "label": "Opération", "type": "select",
         "options": ["ouverture", "fermeture", "gradient", "top-hat", "black-hat"], "default": "ouverture"},
        {"key": "ksize", "label": "Élément", "type": "int", "min": 3, "max": 21, "step": 2, "default": 5}]},
    {"id": "invert", "name": "Inversion", "group": "Autres", "params": []},
]
_DEFAULTS = {s["id"]: {p["key"]: p["default"] for p in s["params"]} for s in SCHEMA}


def apply_pipeline(gray: np.ndarray, pipeline) -> np.ndarray:
    out = gray.copy()
    for step in pipeline or []:
        op = step.get("op")
        if op not in OPS:
            raise ValueError(f"Opération inconnue : {op}")
        params = {**_DEFAULTS[op], **(step.get("params") or {})}
        out = OPS[op](out, params)
    return out


def quality_vs_reference(ref: np.ndarray, img: np.ndarray) -> dict:
    return {
        "psnr": float(min(peak_signal_noise_ratio(ref, img, data_range=255), 99.0)),
        "ssim": float(structural_similarity(ref, img, data_range=255)),
        "mean": float(img.mean()), "std": float(img.std()),
    }
