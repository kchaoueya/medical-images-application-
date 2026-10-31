"""Entraînement d'un U-Net depuis l'application (démarche du TP4) : préparation des données,
séparation apprentissage/validation, entraînement, arrêt anticipé, évaluation IoU/Dice, sauvegarde."""
import json
import re
import tempfile
import threading
import time
import uuid
import zipfile
from pathlib import Path

import cv2
import numpy as np

from . import dl

IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
DATA_ROOT = Path(tempfile.gettempdir()) / "medseg_datasets"
MAX_UNZIPPED = 2 * 1024 ** 3
DATASETS, JOBS = {}, {}
_lock = threading.Lock()


def tf_status():
    try:
        import tensorflow  # noqa: F401
        return {"available": True, "reason": ""}
    except Exception:
        return {"available": False, "reason": "TensorFlow n'est pas installé : pip install tensorflow"}


# ------------------------------------------------------------------ jeu de données
def _safe_extract(zf: zipfile.ZipFile, dest: Path):
    if sum(i.file_size for i in zf.infolist()) > MAX_UNZIPPED:
        raise ValueError("Archive trop volumineuse (> 2 Go décompressée).")
    root = dest.resolve()
    for info in zf.infolist():
        target = (dest / info.filename).resolve()
        if root not in target.parents and target != root:
            raise ValueError("Archive invalide (chemin suspect).")
    zf.extractall(dest)


def collect_pairs(root: Path):
    """Associe images et masques.
    Disposition A (BUSI) : `benign (1).png` + `benign (1)_mask.png` (+ `_mask_1`...) dans le même dossier.
    Disposition B : dossiers `images/` et `masks/` (mêmes noms de fichiers)."""
    files = [p for p in root.rglob("*") if p.suffix.lower() in IMG_EXT and "__MACOSX" not in p.parts]
    pairs = []
    masks_by_img = {}
    plain = []
    for p in files:
        if "_mask" in p.stem:
            masks_by_img.setdefault((p.parent, p.stem.split("_mask")[0]), []).append(p)
        else:
            plain.append(p)
    for p in sorted(plain):
        ms = masks_by_img.get((p.parent, p.stem))
        if ms:
            pairs.append((p, sorted(ms)))
    if pairs:
        return pairs
    mask_dirs = {"masks", "mask", "labels", "label", "gt", "groundtruth"}
    by_stem = {}
    for p in plain:
        if p.parent.name.lower() in mask_dirs:
            by_stem.setdefault(p.stem, []).append(p)
    for p in sorted(plain):
        if p.parent.name.lower() not in mask_dirs and p.stem in by_stem:
            pairs.append((p, by_stem[p.stem]))
    return pairs


def _read_mask(paths, size):
    m = np.zeros((size, size), np.uint8)
    for q in paths:
        a = cv2.imread(str(q), 0)
        if a is not None:
            m = np.maximum(m, cv2.resize(a, (size, size), interpolation=cv2.INTER_NEAREST))
    return (m > 0).astype(np.uint8)


def register_dataset(root: Path, name: str):
    pairs = collect_pairs(root)
    if len(pairs) < 10:
        raise ValueError(f"{len(pairs)} paire(s) image/masque trouvée(s) — il en faut au moins 10. "
                         "Attendu : « nom.png » + « nom_mask.png » dans le même dossier, ou dossiers images/ et masks/.")
    fr = [_read_mask(ms, 96).mean() for _, ms in pairs]
    did = uuid.uuid4().hex[:10]
    DATASETS[did] = {"pairs": pairs, "name": name}
    return {"dataset_id": did, "name": name, "count": len(pairs), "mask_pct": round(100 * float(np.mean(fr)), 2),
            "empty_masks": int(sum(f == 0 for f in fr))}


def dataset_from_zip(data: bytes, name: str):
    d = DATA_ROOT / uuid.uuid4().hex[:8]
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / "up.zip"
    tmp.write_bytes(data)
    with zipfile.ZipFile(tmp) as zf:
        _safe_extract(zf, d / "data")
    tmp.unlink()
    return register_dataset(d / "data", name)


def dataset_from_folder(path: str):
    p = Path(path.strip().strip('"')).expanduser()
    if not p.is_dir():
        raise ValueError("Dossier introuvable sur la machine qui exécute l'application.")
    return register_dataset(p, p.name)


def _normalize(img, norm):
    x = img.astype(np.float32)
    if norm == "tp4_l2":
        n = np.linalg.norm(x, axis=0)
        n[n == 0] = 1
        return x / n
    return x / 255.0


def load_arrays(pairs, size, norm):
    X, Y = [], []
    for ip, ms in pairs:
        a = cv2.imread(str(ip), 0)
        if a is None:
            continue
        X.append(_normalize(cv2.resize(a, (size, size)), norm)[..., None])
        Y.append(_read_mask(ms, size)[..., None].astype(np.float32))
    return np.array(X, np.float32), np.array(Y, np.float32)


# ------------------------------------------------------------------------ métriques
def dice_np(t, p, eps=1e-7):
    t, p = t.astype(bool), p.astype(bool)
    return (2 * np.logical_and(t, p).sum() + eps) / (t.sum() + p.sum() + eps)


def iou_np(t, p, eps=1e-7):
    t, p = t.astype(bool), p.astype(bool)
    return (np.logical_and(t, p).sum() + eps) / (np.logical_or(t, p).sum() + eps)


# ------------------------------------------------------------------------ entraînement
DEFAULT_CFG = {"epochs": 10, "batch_size": 16, "val_split": 0.10, "size": 256, "loss": "bce", "lr": 1e-3,
               "norm": "tp4_l2", "early_stopping": False, "patience": 5, "monitor": "val_loss",
               "augment": False, "dropout_scale": 1.0, "seed": 0, "name": ""}


def start(dataset_id, cfg_in):
    st = tf_status()
    if not st["available"]:
        raise ValueError(st["reason"])
    if dataset_id not in DATASETS:
        raise ValueError("Jeu de données inconnu : rechargez-le.")
    cfg = {**DEFAULT_CFG, **{k: v for k, v in (cfg_in or {}).items() if k in DEFAULT_CFG}}
    cfg["epochs"] = int(np.clip(int(cfg["epochs"]), 1, 300))
    cfg["batch_size"] = int(np.clip(int(cfg["batch_size"]), 1, 64))
    cfg["size"] = int(cfg["size"]) if int(cfg["size"]) in (64, 128, 256) else 256
    cfg["val_split"] = float(np.clip(float(cfg["val_split"]), 0.05, 0.4))
    if cfg["loss"] not in ("bce", "dice", "bce_dice") or cfg["monitor"] not in ("val_loss", "val_dice"):
        raise ValueError("Paramètre invalide.")
    with _lock:
        if any(j["status"] == "running" for j in JOBS.values()):
            raise ValueError("Un entraînement est déjà en cours.")
        jid = uuid.uuid4().hex[:8]
        JOBS[jid] = {"id": jid, "status": "running", "epoch": 0, "epochs": cfg["epochs"], "cfg": cfg, "stop": False,
                     "history": {k: [] for k in ("loss", "val_loss", "accuracy", "val_accuracy", "val_dice", "val_iou")},
                     "message": "Préparation des données…", "metrics": None, "model_id": None, "samples": None,
                     "error": None, "t0": time.time()}
    threading.Thread(target=_run, args=(jid, dataset_id), daemon=True).start()
    return jid


def _run(jid, dataset_id):
    job = JOBS[jid]
    cfg = job["cfg"]
    try:
        import tensorflow as tf
        from sklearn.model_selection import train_test_split
        from .unet import build_unet, get_loss
        tf.keras.utils.set_random_seed(int(cfg["seed"]))
        X, Y = load_arrays(DATASETS[dataset_id]["pairs"], cfg["size"], cfg["norm"])
        Xtr, Xva, Ytr, Yva = train_test_split(X, Y, test_size=cfg["val_split"], random_state=int(cfg["seed"]))
        if cfg["augment"]:
            Xtr, Ytr = np.concatenate([Xtr, Xtr[:, :, ::-1]]), np.concatenate([Ytr, Ytr[:, :, ::-1]])
        job["message"] = f"{len(Xtr)} images d'apprentissage · {len(Xva)} de validation"
        model = build_unet(cfg["size"], cfg["size"], 1, float(cfg["dropout_scale"]))
        model.compile(optimizer=tf.keras.optimizers.Adam(float(cfg["lr"])), loss=get_loss(cfg["loss"]), metrics=["accuracy"])

        class Progress(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                logs = logs if logs is not None else {}
                pred = self.model.predict(Xva, batch_size=32, verbose=0) > 0.5
                logs["val_dice"] = float(np.mean([dice_np(Yva[i], pred[i]) for i in range(len(Xva))]))
                logs["val_iou"] = float(iou_np(Yva, pred))
                for k in job["history"]:
                    if k in logs:
                        job["history"][k].append(float(logs[k]))
                job["epoch"] = epoch + 1
                if job["stop"]:
                    self.model.stop_training = True

        cbs = [Progress()]  # doit précéder EarlyStopping pour que val_dice soit dans `logs`
        if cfg["early_stopping"]:
            cbs.append(tf.keras.callbacks.EarlyStopping(monitor=cfg["monitor"], patience=int(cfg["patience"]),
                                                        mode="max" if cfg["monitor"] == "val_dice" else "min",
                                                        restore_best_weights=True))
        model.fit(Xtr, Ytr, batch_size=cfg["batch_size"], epochs=cfg["epochs"], validation_data=(Xva, Yva),
                  callbacks=cbs, verbose=0)

        loss, acc = model.evaluate(Xva, Yva, verbose=0)
        prob = model.predict(Xva, batch_size=32, verbose=0)
        pred = (prob > 0.5).astype(np.uint8)
        job["metrics"] = {"val_loss": float(loss), "val_accuracy": float(acc), "val_iou": float(iou_np(Yva, pred)),
                          "val_dice": float(np.mean([dice_np(Yva[i], pred[i]) for i in range(len(Xva))])),
                          "epochs_run": len(job["history"]["loss"]),
                          "best_epoch": int(np.argmin(job["history"]["val_loss"])) + 1}
        job["samples"] = _samples(Xva, Yva, pred, cfg["norm"])
        job["model_id"] = _save(model, cfg, job)
        job["status"] = "stopped" if job["stop"] else "done"
        job["message"] = "Entraînement terminé — modèle ajouté à la liste Deep Learning."
    except Exception as e:  # noqa: BLE001
        job["status"], job["error"] = "error", str(e)
        job["message"] = "Échec de l'entraînement."


def _save(model, cfg, job):
    dl.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    base = re.sub(r"[^\w\-]+", "_", cfg["name"].strip()) or \
        f"unet_bs{cfg['batch_size']}_ep{cfg['epochs']}{'_es' if cfg['early_stopping'] else ''}_{cfg['loss']}"
    path = dl.MODELS_DIR / f"{base}.keras"
    if path.exists():
        path = dl.MODELS_DIR / f"{base}_{time.strftime('%H%M%S')}.keras"
    model.save(str(path))
    m = job["metrics"]
    meta = {"name": path.stem, "input_size": cfg["size"], "channels": 1, "norm": cfg["norm"], "layout": "nhwc",
            "threshold": 0.5, "description": f"U-Net entraîné dans MedSeg — Dice val {m['val_dice']:.3f}, IoU {m['val_iou']:.3f}",
            "training": {"cfg": cfg, "history": job["history"], "metrics": m, "date": time.strftime("%Y-%m-%d %H:%M")}}
    path.with_suffix(".json").write_text(json.dumps(meta, indent=1, ensure_ascii=False), encoding="utf-8")
    dl._cache.pop(("keras", str(path)), None)
    return f"file:{path.name}"


def _samples(Xva, Yva, pred, norm, n=6):
    rows = []
    for i in np.linspace(0, len(Xva) - 1, min(n, len(Xva))).astype(int):
        x = Xva[i, :, :, 0]
        img = cv2.normalize(x, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        gt, pr = (Yva[i, :, :, 0] > 0).astype(np.uint8), pred[i, :, :, 0]
        ov = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        for m, col in ((gt, (0, 255, 255)), (pr, (60, 76, 231))):
            c, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(ov, c, -1, col, 2)
        row = np.hstack([cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), cv2.cvtColor(gt * 255, cv2.COLOR_GRAY2BGR),
                         cv2.cvtColor(pr * 255, cv2.COLOR_GRAY2BGR), ov])
        cv2.putText(row, f"Dice {dice_np(gt, pr):.2f}", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 255, 0), 2)
        rows.append(cv2.resize(row, (row.shape[1] // 2, row.shape[0] // 2), interpolation=cv2.INTER_AREA))
    from .io_utils import to_data_url
    return to_data_url(np.vstack(rows))


def status(jid):
    j = JOBS.get(jid)
    if not j:
        raise KeyError("Entraînement inconnu.")
    return {k: v for k, v in j.items() if k != "stop"} | {"elapsed": round(time.time() - j["t0"], 1)}


def stop(jid):
    if jid in JOBS:
        JOBS[jid]["stop"] = True


def list_runs():
    runs = []
    if dl.MODELS_DIR.exists():
        for j in sorted(dl.MODELS_DIR.glob("*.json")):
            try:
                meta = json.loads(j.read_text(encoding="utf-8"))
            except Exception:
                continue
            if "training" in meta:
                runs.append({"id": j.stem, "name": meta.get("name", j.stem), **meta["training"]})
    return runs
