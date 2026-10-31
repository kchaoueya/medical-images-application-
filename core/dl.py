"""Segmentation par apprentissage profond : modèles pré-entraînés.

Trois types de modèles sont gérés :
  * ONNX   (.onnx)                 -> onnxruntime (léger, aucune dépendance lourde)
  * Keras  (.keras .h5 .hdf5)      -> tensorflow (optionnel), p. ex. le U-Net du TP4
  * MedSAM (Hugging Face)          -> torch + transformers (optionnel), invite par boîte englobante

Les fichiers de modèles se déposent dans le dossier `models/`. Un fichier `<nom>.json` du même nom
(facultatif) décrit le prétraitement attendu par le réseau.
"""
import json
import os
import threading
from pathlib import Path

import cv2
import numpy as np

MODELS_DIR = Path(os.environ.get("MEDSEG_MODELS", Path(__file__).resolve().parent.parent / "models"))
MEDSAM_ID = os.environ.get("MEDSEG_MEDSAM_ID", "flaviagiammarino/medsam-vit-base")
EXTS = {".onnx": "onnx", ".keras": "keras", ".h5": "keras", ".hdf5": "keras"}

_cache, _lock = {}, threading.Lock()


# ------------------------------------------------------------------ catalogue
def _sidecar(path: Path) -> dict:
    j = path.with_suffix(".json")
    if j.exists():
        try:
            return json.loads(j.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _module_ok(name):
    try:
        __import__(name)
        return True
    except Exception:
        return False


def list_models():
    out = []
    if MODELS_DIR.exists():
        for f in sorted(MODELS_DIR.iterdir()):
            kind = EXTS.get(f.suffix.lower())
            if not kind:
                continue
            meta = _sidecar(f)
            ok, why = True, ""
            if kind == "keras" and not _module_ok("tensorflow"):
                ok, why = False, "TensorFlow non installé (pip install tensorflow) — ou exporter en ONNX (tools/export_keras_to_onnx.py)"
            out.append({"id": f"file:{f.name}", "name": meta.get("name", f.stem), "kind": kind, "available": ok,
                        "reason": why, "description": meta.get("description", f"Modèle {kind.upper()} — {f.name}"),
                        "needs_seed": False})
    ok = _module_ok("torch") and _module_ok("transformers")
    out.append({"id": "medsam", "name": "MedSAM (ViT-B)", "kind": "medsam", "available": ok,
                "reason": "" if ok else "pip install torch transformers (poids téléchargés depuis Hugging Face au 1er usage)",
                "description": "Modèle fondation de segmentation médicale ; invite = boîte englobante (cercle du germe).",
                "needs_seed": True})
    return out


# ------------------------------------------------------------- prétraitement
def _normalize(img, mode):
    x = img.astype(np.float32)
    if mode == "tp4_l2":  # tf.keras.utils.normalize(axis=1) du TP4 : norme L2 le long de la hauteur
        n = np.linalg.norm(x, axis=0)
        n[n == 0] = 1
        return x / n
    if mode == "zscore":
        return (x - x.mean()) / (x.std() + 1e-7)
    if mode == "imagenet":
        x = x / 255.0
        return x  # la moyenne/écart-type par canal est appliquée dans _prep
    return x / 255.0


def _prep(gray, size, channels, norm):
    x = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA if gray.shape[0] > size else cv2.INTER_LINEAR)
    x = _normalize(x, norm)
    if channels == 3:
        x = np.stack([x] * 3, -1)
        if norm == "imagenet":
            x = (x - np.array([0.485, 0.456, 0.406], np.float32)) / np.array([0.229, 0.224, 0.225], np.float32)
    else:
        x = x[..., None]
    return x.astype(np.float32)  # (H, W, C)


def _to_prob(arr, layout, channel):
    """arr : sortie sans batch -> carte 2D en [0,1]."""
    if arr.ndim == 3:
        if layout == "nchw":
            arr = arr[min(channel, arr.shape[0] - 1)]
        else:
            arr = arr[..., min(channel, arr.shape[-1] - 1)]
    arr = np.squeeze(arr).astype(np.float32)
    if arr.min() < 0 or arr.max() > 1:  # logits -> sigmoïde
        arr = 1.0 / (1.0 + np.exp(-np.clip(arr, -30, 30)))
    return arr


# --------------------------------------------------------------------- ONNX
def _predict_onnx(path, gray, meta):
    import onnxruntime as ort
    with _lock:
        sess = _cache.get(("onnx", str(path)))
        if sess is None:
            sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
            _cache[("onnx", str(path))] = sess
    inp = sess.get_inputs()[0]
    shp = inp.shape
    layout = meta.get("layout")
    if layout is None:
        layout = "nchw" if (len(shp) == 4 and shp[1] in (1, 3) and shp[-1] not in (1, 3)) else "nhwc"
    spatial = [d for d in (shp[2:] if layout == "nchw" else shp[1:3]) if isinstance(d, int)]
    size = int(meta.get("input_size", spatial[0] if spatial else 256))
    ch_dim = shp[1] if layout == "nchw" else shp[-1]
    channels = int(meta.get("channels", ch_dim if isinstance(ch_dim, int) else 1))
    x = _prep(gray, size, channels, meta.get("norm", "unit"))
    x = x.transpose(2, 0, 1)[None] if layout == "nchw" else x[None]
    out = sess.run(None, {inp.name: x.astype(np.float32)})[0][0]
    return _to_prob(out, layout, int(meta.get("output_channel", 0)))


# --------------------------------------------------------------------- Keras
def _predict_keras(path, gray, meta):
    import tensorflow as tf
    with _lock:
        model = _cache.get(("keras", str(path)))
        if model is None:
            model = tf.keras.models.load_model(str(path), compile=False)
            _cache[("keras", str(path))] = model
    _, h, w, c = model.input_shape
    norm = meta.get("norm") or ("tp4_l2" if path.name.lower().startswith("unet") else "unit")
    x = _prep(gray, int(meta.get("input_size", h or 256)), int(meta.get("channels", c or 1)), norm)
    out = model.predict(x[None], verbose=0)[0]
    return _to_prob(out, "nhwc", int(meta.get("output_channel", 0)))


# -------------------------------------------------------------------- MedSAM
def _predict_medsam(gray, box_xyxy):
    import torch
    from PIL import Image
    from transformers import SamModel, SamProcessor
    with _lock:
        if "medsam" not in _cache:
            _cache["medsam"] = (SamModel.from_pretrained(MEDSAM_ID).eval(), SamProcessor.from_pretrained(MEDSAM_ID))
    model, proc = _cache["medsam"]
    rgb = Image.fromarray(cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB))
    inputs = proc(rgb, input_boxes=[[list(map(float, box_xyxy))]], return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs, multimask_output=False)
    probs = proc.post_process_masks(outputs.pred_masks.sigmoid().cpu(), inputs["original_sizes"].cpu(),
                                    inputs["reshaped_input_sizes"].cpu(), binarize=False)[0]
    return np.squeeze(probs.numpy()).astype(np.float32)


# ------------------------------------------------------------------- façade
def predict(model_id, gray, threshold=0.5, seed_box=None):
    """Retourne (masque binaire uint8, carte de probabilité float32 [0,1])."""
    if model_id == "medsam":
        if seed_box is None:
            raise ValueError("MedSAM nécessite un germe (boîte englobante).")
        prob = _predict_medsam(gray, seed_box)
    else:
        name = model_id.split(":", 1)[1]
        path = (MODELS_DIR / name).resolve()
        if path.parent != MODELS_DIR.resolve() or not path.exists():
            raise ValueError("Modèle introuvable.")
        meta = _sidecar(path)
        kind = EXTS[path.suffix.lower()]
        prob = _predict_onnx(path, gray, meta) if kind == "onnx" else _predict_keras(path, gray, meta)
        threshold = float(meta.get("threshold", threshold)) if threshold is None else threshold
    prob = cv2.resize(prob, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_LINEAR)
    return (prob > threshold).astype(np.uint8), prob
