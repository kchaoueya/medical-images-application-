"""MedSeg — plateforme web d'analyse et de segmentation d'images médicales."""
import io
import json
import time
import uuid
import zipfile
from collections import OrderedDict

import cv2
import numpy as np
from flask import Flask, jsonify, request, send_file, send_from_directory

from core import dl, io_utils as iou_, metrics, preprocess, segmentation, training

app = Flask(__name__, static_folder="static", static_url_path="")
app.config["MAX_CONTENT_LENGTH"] = 1024 * 1024 * 1024  # archives de jeux de données

STORE = OrderedDict()   # image_id -> {"gray", "gt", "name", "results"}
MAX_IMAGES = 20


def _get(image_id):
    if image_id not in STORE:
        raise KeyError("Image inconnue ou expirée : réimportez-la.")
    return STORE[image_id]


@app.errorhandler(Exception)
def _err(e):
    code = 404 if isinstance(e, KeyError) else 400 if isinstance(e, ValueError) else 500
    if code == 500:
        app.logger.exception(e)
    return jsonify(error=str(e.args[0] if isinstance(e, KeyError) else e)), code


@app.get("/")
def index():
    return send_from_directory("static", "index.html")


@app.get("/samples/<path:name>")
def sample_file(name):
    return send_from_directory("samples", name)


@app.get("/api/samples")
def samples():
    import os
    names = sorted(f[:-4] for f in os.listdir("samples") if f.endswith(".png") and not f.endswith("_mask.png"))
    return jsonify(samples=[{"name": n, "image": f"/samples/{n}.png", "mask": f"/samples/{n}_mask.png"} for n in names])


@app.get("/api/schema")
def schema():
    return jsonify(preprocess=preprocess.SCHEMA, segmentation=segmentation.SCHEMA, models=dl.list_models())


@app.post("/api/upload")
def upload():
    f = request.files.get("image")
    if f is None:
        raise ValueError("Aucun fichier reçu.")
    gray = iou_.decode_gray(f.read())
    gt = None
    if request.files.get("mask"):
        gt = iou_.decode_mask(request.files["mask"].read(), gray.shape)
    iid = uuid.uuid4().hex[:12]
    STORE[iid] = {"gray": gray, "gt": gt, "name": f.filename or "image", "results": {}}
    while len(STORE) > MAX_IMAGES:
        STORE.popitem(last=False)
    return jsonify(image_id=iid, name=STORE[iid]["name"], width=gray.shape[1], height=gray.shape[0],
                   image=iou_.to_data_url(gray), has_gt=gt is not None)


@app.post("/api/mask/<image_id>")
def set_mask(image_id):
    d = _get(image_id)
    d["gt"] = iou_.decode_mask(request.files["mask"].read(), d["gray"].shape)
    return jsonify(ok=True)


def _hist(g):
    return np.bincount(g.ravel(), minlength=256).tolist()


@app.post("/api/preprocess")
def do_preprocess():
    j = request.get_json(force=True)
    d = _get(j["image_id"])
    out = preprocess.apply_pipeline(d["gray"], j.get("pipeline"))
    return jsonify(image=iou_.to_data_url(out), quality=preprocess.quality_vs_reference(d["gray"], out),
                   hist_before=_hist(d["gray"]), hist_after=_hist(out))


@app.post("/api/segment")
def do_segment():
    j = request.get_json(force=True)
    d = _get(j["image_id"])
    gray = preprocess.apply_pipeline(d["gray"], j.get("pipeline"))
    ctx = {"polarity": j.get("polarity", "dark"), "post": j.get("post", {}), "seed": j.get("seed", {})}
    h, w = gray.shape
    cx, cy, r = segmentation._seed(ctx, gray.shape)
    box = (max(cx - r, 0), max(cy - r, 0), min(cx + r, w - 1), min(cy + r, h - 1))
    names = {s["id"]: s["name"] for s in segmentation.SCHEMA}
    names.update({m["id"]: m["name"] for m in dl.list_models()})

    results, masks = [], {}
    for k, item in enumerate(j.get("methods", [])):
        mid, params = item["id"], item.get("params", {})
        is_dl = mid == "medsam" or mid.startswith("file:")
        res = {"id": mid, "name": names.get(mid, mid), "family": "dl" if is_dl else "conv"}
        t0 = time.perf_counter()
        try:
            if is_dl:
                mask, prob = dl.predict(mid, gray, float(params.get("threshold", .5)), box)
                mask = segmentation.postprocess(mask, gray, ctx["post"]) if ctx["post"].get("apply_to_dl") else mask
                info = {"seuil proba": float(params.get("threshold", .5))}
            else:
                mask, info = segmentation.run(mid, gray, params, ctx)
            res["time_ms"] = round((time.perf_counter() - t0) * 1000, 1)
            res["info"] = info
            res["stats"] = metrics.shape_stats(mask, gray)
            col = iou_.PALETTE[len(masks) % len(iou_.PALETTE)]
            res["color"] = iou_.PALETTE_HEX[len(masks) % len(iou_.PALETTE)]
            res["overlay"] = iou_.to_data_url(iou_.overlay(gray, mask, col))
            res["mask"] = iou_.to_data_url(mask * 255)
            if d["gt"] is not None:
                res["vs_gt"] = {"dice": metrics.dice(mask, d["gt"]), "iou": metrics.iou(mask, d["gt"]),
                                "hausdorff": metrics.hausdorff(mask, d["gt"])}
            masks[mid] = mask
        except Exception as e:  # une méthode en échec ne bloque pas les autres
            res["error"] = str(e)
        results.append(res)

    d["results"] = {"masks": masks, "gray": gray, "names": names, "results": results}
    return jsonify(results=results, pairwise=metrics.pairwise(masks) if len(masks) > 1 else None,
                   has_gt=d["gt"] is not None)


@app.post("/api/compare")
def compare():
    """Image avec les contours superposés de plusieurs méthodes (et de la référence si disponible)."""
    j = request.get_json(force=True)
    d = _get(j["image_id"])
    cache = d.get("results") or {}
    masks = cache.get("masks", {})
    sel = [m for m in j.get("methods", []) if m in masks]
    ids = list(masks)
    pairs = [(masks[m], iou_.PALETTE[ids.index(m) % len(iou_.PALETTE)]) for m in sel]
    img = iou_.contours_overlay(cache["gray"], pairs, d["gt"] if j.get("show_gt", True) else None)
    return jsonify(image=iou_.to_data_url(img))


# ------------------------------------------------------------ entraînement U-Net (TP4)
@app.get("/api/train/info")
def train_info():
    return jsonify(**training.tf_status(), defaults=training.DEFAULT_CFG, runs=training.list_runs())


@app.post("/api/train/dataset")
def train_dataset():
    if request.files.get("zip"):
        f = request.files["zip"]
        return jsonify(training.dataset_from_zip(f.read(), f.filename or "archive"))
    path = (request.form.get("path") or "").strip()
    if path:
        if request.remote_addr not in ("127.0.0.1", "::1"):
            raise ValueError("Le chargement par chemin local n'est autorisé que depuis la machine locale.")
        return jsonify(training.dataset_from_folder(path))
    raise ValueError("Fournissez une archive .zip ou un chemin de dossier.")


@app.post("/api/train/start")
def train_start():
    j = request.get_json(force=True)
    return jsonify(job_id=training.start(j.get("dataset_id"), j.get("cfg")))


@app.get("/api/train/status/<job_id>")
def train_status(job_id):
    return jsonify(training.status(job_id))


@app.post("/api/train/stop/<job_id>")
def train_stop(job_id):
    training.stop(job_id)
    return jsonify(ok=True)


@app.get("/api/export/<image_id>")
def export(image_id):
    d = _get(image_id)
    cache = d.get("results") or {}
    if not cache:
        raise ValueError("Aucun résultat à exporter : lancez d'abord une segmentation.")
    buf = io.BytesIO()
    rows = ["methode;temps_ms;aire_px;aire_pct;perimetre;circularite;dice_ref;iou_ref;hausdorff_ref"]
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("image_pretraitee.png", iou_.png_bytes(cache["gray"]))
        for r in cache["results"]:
            if "error" in r:
                continue
            safe = r["id"].replace(":", "_").replace("/", "_")
            m = cache["masks"][r["id"]]
            z.writestr(f"masques/{safe}.png", iou_.png_bytes(m * 255))
            z.writestr(f"superpositions/{safe}.png",
                       iou_.png_bytes(iou_.overlay(cache["gray"], m)))
            s, g = r["stats"], r.get("vs_gt", {})
            rows.append(";".join(str(x) for x in [r["name"], r["time_ms"], s["area_px"], s["area_pct"], s["perimeter"],
                                                  s["circularity"], g.get("dice", ""), g.get("iou", ""), g.get("hausdorff", "")]))
        z.writestr("resultats.csv", "\n".join(rows))
    buf.seek(0)
    return send_file(buf, mimetype="application/zip", as_attachment=True, download_name="medseg_resultats.zip")


if __name__ == "__main__":
    import os
    app.run(host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", 5000)), debug=False)
