import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from core import dl, metrics, preprocess, segmentation  # noqa: E402
import app as appmod  # noqa: E402

S = ROOT / "samples"
CASES = {"echo_lesion": "dark", "irm_tumeur": "bright", "endoscopie_glotte": "dark"}
POST = {"exclude_bg": True, "bg_thr": 15, "open_k": 3, "largest": True, "fill_holes": True, "min_area": 300}
SEED = {"cx": .5, "cy": .5, "r": .15}


def load(n):
    g = cv2.imread(str(S / f"{n}.png"), 0)
    m = (cv2.imread(str(S / f"{n}_mask.png"), 0) > 127).astype(np.uint8)
    return g, m


@pytest.mark.parametrize("method", list(segmentation.METHODS))
@pytest.mark.parametrize("case", list(CASES))
def test_every_method_runs(method, case):
    g, _ = load(case)
    m, _ = segmentation.run(method, g, {}, {"polarity": CASES[case], "post": POST, "seed": SEED})
    assert m.shape == g.shape and set(np.unique(m)) <= {0, 1}


@pytest.mark.parametrize("case,method,minimum", [
    ("echo_lesion", "kmeans", .85), ("irm_tumeur", "kmeans", .9),
    ("irm_tumeur", "multiotsu", .9),
    ("endoscopie_glotte", "kmeans", .8), ("echo_lesion", "regiongrowing", .5)])
def test_quality(case, method, minimum):
    g, gt = load(case)
    g = preprocess.apply_pipeline(g, [{"op": "median", "params": {"ksize": 5}}])
    m, _ = segmentation.run(method, g, {}, {"polarity": CASES[case], "post": POST, "seed": SEED})
    assert metrics.dice(m, gt) >= minimum


def test_otsu_limit_two_classes():
    """Comme observé au TP2/TP3 : Otsu (2 classes) sépare tête/fond, pas la tumeur de l'IRM."""
    g, gt = load("irm_tumeur")
    m, _ = segmentation.run("otsu", g, {}, {"polarity": "bright", "post": POST, "seed": SEED})
    assert metrics.dice(m, gt) < .3


def test_metrics():
    a = np.zeros((10, 10), np.uint8)
    a[2:6, 2:6] = 1
    assert metrics.dice(a, a) == pytest.approx(1) and metrics.iou(a, a) == pytest.approx(1)
    b = np.zeros_like(a)
    b[4:8, 4:8] = 1
    assert metrics.dice(a, b) == pytest.approx(2 * 4 / 32, abs=1e-3)


def test_preprocess_all_ops():
    g, _ = load("echo_lesion")
    for s in preprocess.SCHEMA:
        out = preprocess.apply_pipeline(g, [{"op": s["id"]}])
        assert out.shape == g.shape and out.dtype == np.uint8


def _tiny_onnx(path):
    """Modèle ONNX factice (flou + sigmoïde) : valide la chaîne DL, sans valeur médicale."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper
    w = numpy_helper.from_array(np.ones((1, 1, 5, 5), np.float32) / 25, "w")
    b = numpy_helper.from_array(np.array([-0.4], np.float32), "b")
    nodes = [helper.make_node("Conv", ["x", "w", "b"], ["c"], pads=[2, 2, 2, 2]),
             helper.make_node("Sigmoid", ["c"], ["y"])]
    inp = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1, 128, 128])
    out = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 1, 128, 128])
    g = helper.make_graph(nodes, "t", [inp], [out], [w, b])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])
    m.ir_version = 8
    onnx.save(m, str(path))


def test_onnx_pipeline(tmp_path, monkeypatch):
    pytest.importorskip("onnx")
    _tiny_onnx(tmp_path / "tiny.onnx")
    (tmp_path / "tiny.json").write_text(json.dumps({"name": "Tiny", "input_size": 128, "norm": "unit"}))
    monkeypatch.setattr(dl, "MODELS_DIR", tmp_path)
    assert any(m["id"] == "file:tiny.onnx" and m["available"] for m in dl.list_models())
    g, _ = load("irm_tumeur")
    mask, prob = dl.predict("file:tiny.onnx", g, .5)
    assert mask.shape == g.shape and 0 <= prob.min() and prob.max() <= 1 and mask.any()


def test_tp4_normalisation_matches_keras():
    g, _ = load("echo_lesion")
    small = cv2.resize(g, (256, 256))
    x = dl._normalize(small, "tp4_l2")
    ref = small.astype(np.float32)
    ref = ref / np.expand_dims(np.linalg.norm(ref, axis=0), 0)
    assert np.allclose(x, ref)


def test_model_path_traversal_blocked():
    g, _ = load("echo_lesion")
    with pytest.raises(ValueError):
        dl.predict("file:../app.py", g, .5)


def test_api_end_to_end():
    c = appmod.app.test_client()
    assert c.get("/api/schema").status_code == 200
    r = c.post("/api/upload", data={"image": (open(S / "irm_tumeur.png", "rb"), "irm.png"),
                                    "mask": (open(S / "irm_tumeur_mask.png", "rb"), "m.png")},
               content_type="multipart/form-data").get_json()
    iid = r["image_id"]
    assert r["has_gt"]
    pipe = [{"op": "median", "params": {"ksize": 5}}]
    p = c.post("/api/preprocess", json={"image_id": iid, "pipeline": pipe}).get_json()
    assert 0 < p["quality"]["ssim"] <= 1 and len(p["hist_after"]) == 256
    seg = c.post("/api/segment", json={
        "image_id": iid, "polarity": "bright", "post": POST, "seed": SEED, "pipeline": pipe,
        "methods": [{"id": "otsu"}, {"id": "kmeans"}, {"id": "gac"}, {"id": "medsam"}]}).get_json()
    ok = [x for x in seg["results"] if "error" not in x]
    assert len(ok) >= 3 and seg["pairwise"] and all("vs_gt" in x for x in ok)
    cmp_ = c.post("/api/compare", json={"image_id": iid, "methods": ["otsu", "kmeans"]}).get_json()
    assert cmp_["image"].startswith("data:image/png")
    z = c.get(f"/api/export/{iid}")
    assert z.status_code == 200 and z.data[:2] == b"PK"
    assert c.post("/api/segment", json={"image_id": "nope", "methods": []}).status_code == 404


def test_dataset_pairing_and_zip_safety(tmp_path):
    import io, zipfile
    from core import training
    sys.path.insert(0, str(ROOT / "tests"))
    import make_toy_dataset
    make_toy_dataset.make(tmp_path / "ds", n=12, S=96)
    info = training.dataset_from_folder(str(tmp_path / "ds"))
    assert info["count"] == 12 and 0 < info["mask_pct"] < 50
    evil = io.BytesIO()
    with zipfile.ZipFile(evil, "w") as z:
        z.writestr("../../evil.png", b"x")
    with pytest.raises(ValueError):
        training.dataset_from_zip(evil.getvalue(), "evil")


def test_training_end_to_end(tmp_path, monkeypatch):
    pytest.importorskip("tensorflow")
    import time
    from core import training
    sys.path.insert(0, str(ROOT / "tests"))
    import make_toy_dataset
    make_toy_dataset.make(tmp_path / "ds", n=30, S=96)
    monkeypatch.setattr(dl, "MODELS_DIR", tmp_path / "models")
    did = training.dataset_from_folder(str(tmp_path / "ds"))["dataset_id"]
    jid = training.start(did, {"epochs": 3, "size": 64, "batch_size": 8, "early_stopping": True, "patience": 2,
                               "monitor": "val_dice", "name": "t"})
    while training.status(jid)["status"] == "running":
        time.sleep(1)
    s = training.status(jid)
    assert s["status"] == "done", s["error"]
    assert len(s["history"]["val_dice"]) == s["metrics"]["epochs_run"] <= 3
    assert s["model_id"] == "file:t.keras" and any(r["id"] == "t" for r in training.list_runs())
    g, _ = load("echo_lesion")
    mask, _ = dl.predict(s["model_id"], g, .5)
    assert mask.shape == g.shape
