"""Méthodes de segmentation conventionnelles (reprend les approches des TP2 et TP3)."""
import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage import segmentation as seg
from skimage.filters import threshold_multiotsu
from sklearn.cluster import KMeans, MeanShift, estimate_bandwidth

WORK_SIZE = 256  # taille de travail des méthodes itératives (Chan-Vese, contours actifs)


# ---------------------------------------------------------------- utilitaires
def _odd(v, lo=1):
    v = max(int(v), lo)
    return v if v % 2 else v + 1


def _downscale(gray, size):
    h, w = gray.shape
    s = size / max(h, w)
    if s >= 1:
        return gray, 1.0
    return cv2.resize(gray, (max(int(w * s), 8), max(int(h * s), 8)), interpolation=cv2.INTER_AREA), s


def _up(arr, shape, nearest=True):
    return cv2.resize(arr, (shape[1], shape[0]),
                      interpolation=cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR)


def background_mask(gray, thr):
    """Pixels sombres (<= thr) connectés aux bords de l'image : le fond noir (TP3)."""
    n, lab = cv2.connectedComponents((gray <= thr).astype(np.uint8))
    border = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))) - {0}
    return np.isin(lab, list(border)) if border else np.zeros(gray.shape, bool)


def select_region(labels, gray, polarity, bg, min_area):
    """Parmi les régions d'une partition, choisit la plus sombre ('dark') ou la plus claire ('bright'),
    hors fond et hors régions trop petites (démarche du TP3)."""
    best, best_mean = None, None
    for l in np.unique(labels):
        reg = (labels == l) & ~bg
        if reg.sum() < min_area:
            continue
        m = gray[reg].mean()
        if best is None or (m < best_mean if polarity == "dark" else m > best_mean):
            best, best_mean = reg, m
    if best is None:  # repli : on ignore la contrainte de surface
        for l in np.unique(labels):
            reg = (labels == l) & ~bg
            if reg.sum() == 0:
                continue
            m = gray[reg].mean()
            if best is None or (m < best_mean if polarity == "dark" else m > best_mean):
                best, best_mean = reg, m
    return (best if best is not None else np.zeros(gray.shape, bool)).astype(np.uint8)


def postprocess(mask, gray, post):
    """Nettoyage commun : exclusion du fond, ouverture, plus grande composante, remplissage, surface min."""
    m = mask.astype(np.uint8)
    if post.get("exclude_bg"):
        m[background_mask(gray, int(post.get("bg_thr", 15)))] = 0
    k = int(post.get("open_k", 3))
    if k > 1:
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    if post.get("fill_holes", True):
        m = ndi.binary_fill_holes(m).astype(np.uint8)
    if post.get("largest", True) and m.any():
        n, lab, stats, _ = cv2.connectedComponentsWithStats(m)
        if n > 2:
            m = (lab == 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])).astype(np.uint8)
    ma = int(post.get("min_area", 0))
    if ma > 0 and m.sum() < ma:
        m[:] = 0
    return m


def _seed(ctx, shape):
    s = ctx.get("seed", {})
    h, w = shape
    return (int(np.clip(s.get("cx", .5), 0, 1) * (w - 1)), int(np.clip(s.get("cy", .5), 0, 1) * (h - 1)),
            max(int(s.get("r", .15) * min(h, w)), 3))


def _bg(gray, ctx):
    p = ctx.get("post", {})
    return background_mask(gray, int(p.get("bg_thr", 15))) if p.get("exclude_bg") else np.zeros(gray.shape, bool)


def _min_area(gray, ctx):
    return max(int(ctx.get("post", {}).get("min_area", 0)), 20)


# ------------------------------------------------------------------ méthodes
def otsu(gray, p, ctx):
    t, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m = gray < t if ctx["polarity"] == "dark" else gray > t
    return m.astype(np.uint8), {"seuil": float(t)}


def manual_threshold(gray, p, ctx):
    t = float(p["t"])
    m = gray < t if ctx["polarity"] == "dark" else gray > t
    return m.astype(np.uint8), {"seuil": t}


def adaptive_threshold(gray, p, ctx):
    mode = cv2.THRESH_BINARY_INV if ctx["polarity"] == "dark" else cv2.THRESH_BINARY
    m = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, mode, _odd(p["block"], 3), float(p["C"]))
    return (m > 0).astype(np.uint8), {}


def multi_otsu(gray, p, ctx):
    k = int(p["classes"])
    th = threshold_multiotsu(gray, classes=k)
    labels = np.digitize(gray, bins=th)
    m = select_region(labels, gray, ctx["polarity"], _bg(gray, ctx), _min_area(gray, ctx))
    return m, {"seuils": [round(float(t), 1) for t in th]}


def kmeans(gray, p, ctx):
    k = int(p["k"])
    x = gray.reshape(-1, 1).astype(np.float32)
    rng = np.random.default_rng(0)
    sample = x[rng.choice(len(x), min(len(x), 30000), replace=False)]
    km = KMeans(n_clusters=k, n_init=3, random_state=0).fit(sample)
    labels = km.predict(x).reshape(gray.shape)
    m = select_region(labels, gray, ctx["polarity"], _bg(gray, ctx), _min_area(gray, ctx))
    return m, {"centres": sorted(round(float(c), 1) for c in km.cluster_centers_.ravel())}


def meanshift(gray, p, ctx):
    small, _ = _downscale(gray, int(p["size"]))
    small = cv2.medianBlur(small, 5)
    x = small.reshape(-1, 1).astype(np.float64)
    bw = estimate_bandwidth(x, quantile=float(p["quantile"]), n_samples=500, random_state=0)
    bw = max(bw, float(p["bw_min"]))
    ms = MeanShift(bandwidth=bw, bin_seeding=True).fit(x)
    labels = _up(ms.labels_.reshape(small.shape).astype(np.int32), gray.shape).astype(np.int32)
    m = select_region(labels, gray, ctx["polarity"], _bg(gray, ctx), _min_area(gray, ctx))
    return m, {"bande passante": round(float(bw), 2), "régions": int(labels.max() + 1)}


def chan_vese(gray, p, ctx):
    small, _ = _downscale(gray, WORK_SIZE)
    img = small.astype(np.float64) / 255.0
    init = p["init"]
    if init == "disque (germe)":
        cx, cy, r = _seed(ctx, small.shape)
        init_ls = np.zeros(small.shape, np.int8)
        cv2.circle(init_ls, (cx, cy), r, 1, -1)
    else:
        init_ls = "checkerboard"
    ls = seg.morphological_chan_vese(img, num_iter=int(p["iterations"]), init_level_set=init_ls,
                                     smoothing=int(p["smoothing"]), lambda1=float(p["lambda1"]),
                                     lambda2=float(p["lambda2"]))
    labels = _up(ls.astype(np.uint8), gray.shape)
    m = select_region(labels, gray, ctx["polarity"], _bg(gray, ctx), _min_area(gray, ctx))
    return m, {}


def active_contour(gray, p, ctx):
    """Contour actif géodésique morphologique (GAC) initialisé par le germe (disque)."""
    small, s = _downscale(gray, WORK_SIZE)
    img = cv2.GaussianBlur(small, (0, 0), 1).astype(np.float64) / 255.0
    g = seg.inverse_gaussian_gradient(img, alpha=float(p["alpha"]), sigma=float(p["sigma"]))
    cx, cy, r = _seed(ctx, small.shape)
    init = np.zeros(small.shape, np.int8)
    cv2.circle(init, (cx, cy), r, 1, -1)
    ls = seg.morphological_geodesic_active_contour(
        g, num_iter=int(p["iterations"]), init_level_set=init, smoothing=int(p["smoothing"]),
        balloon=float(p["balloon"]), threshold="auto")
    return _up(ls.astype(np.uint8), gray.shape), {}


def watershed(gray, p, ctx):
    """Ligne de partage des eaux par marqueurs sur l'image gradient."""
    sm = cv2.GaussianBlur(gray, (0, 0), float(p["sigma"]))
    grad = cv2.morphologyEx(sm, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)).astype(np.float64)
    markers = np.zeros(gray.shape, np.int32)
    if p["markers"] == "germe":
        cx, cy, r = _seed(ctx, gray.shape)
        yy, xx = np.ogrid[:gray.shape[0], :gray.shape[1]]
        d2 = (xx - cx) ** 2 + (yy - cy) ** 2
        markers[d2 > (r * 2.5) ** 2] = 1                   # fond : hors du disque 2,5 r
        markers[d2 <= (max(r // 2, 2)) ** 2] = 2           # objet : petit disque central
    else:  # automatique : Otsu + distance transform (méthode classique)
        t, _ = cv2.threshold(sm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        obj = (sm < t) if ctx["polarity"] == "dark" else (sm > t)
        obj = cv2.morphologyEx(obj.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        dist = cv2.distanceTransform(obj, cv2.DIST_L2, 5)
        sure_fg = dist > float(p["fg_ratio"]) * dist.max() if dist.max() > 0 else obj > 0
        sure_bg = ~cv2.dilate(obj, np.ones((3, 3), np.uint8), iterations=3).astype(bool)
        markers[sure_bg] = 1
        markers[sure_fg] = 2
    lab = seg.watershed(grad, markers)
    return (lab == 2).astype(np.uint8), {}


def region_growing(gray, p, ctx):
    cx, cy, _ = _seed(ctx, gray.shape)
    sm = cv2.GaussianBlur(gray, (0, 0), float(p["sigma"]))
    m = seg.flood(sm, (cy, cx), tolerance=float(p["tolerance"]))
    return m.astype(np.uint8), {"graine": [cx, cy]}


METHODS = {
    "otsu": otsu, "threshold": manual_threshold, "adaptive": adaptive_threshold, "multiotsu": multi_otsu,
    "kmeans": kmeans, "meanshift": meanshift, "chanvese": chan_vese, "gac": active_contour,
    "watershed": watershed, "regiongrowing": region_growing,
}

SCHEMA = [
    {"id": "otsu", "name": "Seuillage d'Otsu", "group": "Seuillage", "params": []},
    {"id": "threshold", "name": "Seuillage manuel", "group": "Seuillage", "params": [
        {"key": "t", "label": "Seuil", "type": "int", "min": 0, "max": 255, "step": 1, "default": 100}]},
    {"id": "adaptive", "name": "Seuillage adaptatif", "group": "Seuillage", "params": [
        {"key": "block", "label": "Bloc", "type": "int", "min": 5, "max": 101, "step": 2, "default": 31},
        {"key": "C", "label": "Constante C", "type": "float", "min": -20, "max": 20, "step": 1, "default": 5}]},
    {"id": "multiotsu", "name": "Multi-Otsu", "group": "Seuillage", "params": [
        {"key": "classes", "label": "Classes", "type": "int", "min": 2, "max": 5, "step": 1, "default": 3}]},
    {"id": "kmeans", "name": "K-means", "group": "Classification de pixels", "params": [
        {"key": "k", "label": "k", "type": "int", "min": 2, "max": 8, "step": 1, "default": 3}]},
    {"id": "meanshift", "name": "Mean-shift", "group": "Classification de pixels", "params": [
        {"key": "quantile", "label": "Quantile (bande passante)", "type": "float", "min": .05, "max": .6, "step": .05, "default": .2},
        {"key": "bw_min", "label": "Bande passante min.", "type": "float", "min": 1, "max": 40, "step": 1, "default": 8},
        {"key": "size", "label": "Taille de travail", "type": "int", "min": 64, "max": 256, "step": 32, "default": 128}]},
    {"id": "chanvese", "name": "Level-set (Chan-Vese)", "group": "Contours actifs", "params": [
        {"key": "iterations", "label": "Itérations", "type": "int", "min": 10, "max": 300, "step": 10, "default": 100},
        {"key": "smoothing", "label": "Lissage", "type": "int", "min": 0, "max": 5, "step": 1, "default": 1},
        {"key": "lambda1", "label": "λ1 (intérieur)", "type": "float", "min": .5, "max": 3, "step": .5, "default": 1},
        {"key": "lambda2", "label": "λ2 (extérieur)", "type": "float", "min": .5, "max": 3, "step": .5, "default": 1},
        {"key": "init", "label": "Initialisation", "type": "select", "options": ["damier", "disque (germe)"], "default": "damier"}]},
    {"id": "gac", "name": "Contour actif géodésique", "group": "Contours actifs", "needs_seed": True, "params": [
        {"key": "iterations", "label": "Itérations", "type": "int", "min": 20, "max": 400, "step": 10, "default": 150},
        {"key": "smoothing", "label": "Lissage", "type": "int", "min": 0, "max": 5, "step": 1, "default": 2},
        {"key": "balloon", "label": "Ballon (+ gonfle, − rétrécit)", "type": "float", "min": -1, "max": 1, "step": .1, "default": 1},
        {"key": "alpha", "label": "α (sensibilité gradient)", "type": "float", "min": 10, "max": 300, "step": 10, "default": 100},
        {"key": "sigma", "label": "σ gradient", "type": "float", "min": 1, "max": 8, "step": .5, "default": 3}]},
    {"id": "watershed", "name": "Ligne de partage des eaux", "group": "Régions", "needs_seed": True, "params": [
        {"key": "markers", "label": "Marqueurs", "type": "select", "options": ["automatiques", "germe"], "default": "automatiques"},
        {"key": "sigma", "label": "Lissage σ", "type": "float", "min": .5, "max": 5, "step": .5, "default": 1.5},
        {"key": "fg_ratio", "label": "Seuil distance (auto)", "type": "float", "min": .1, "max": .9, "step": .05, "default": .5}]},
    {"id": "regiongrowing", "name": "Croissance de région", "group": "Régions", "needs_seed": True, "params": [
        {"key": "tolerance", "label": "Tolérance (niveaux de gris)", "type": "float", "min": 2, "max": 80, "step": 1, "default": 20},
        {"key": "sigma", "label": "Lissage σ", "type": "float", "min": 0, "max": 4, "step": .5, "default": 1.5}]},
]
_DEFAULTS = {s["id"]: {p["key"]: p["default"] for p in s["params"]} for s in SCHEMA}


def run(method, gray, params, ctx):
    p = {**_DEFAULTS[method], **(params or {})}
    raw, info = METHODS[method](gray, p, ctx)
    return postprocess(raw, gray, ctx.get("post", {})), info
