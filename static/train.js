"use strict";
/* Onglet « Entraîner un U-Net » — reprend la démarche du TP4 (U-Net, 10 vs 50 époques, arrêt anticipé, Dice/IoU). */
const T = { ds: null, job: null, timer: null, runs: [], sel: new Set(), metric: "val_dice", cfg: {} };
const COLORS = ["#e74c3c", "#3498db", "#2ecc71", "#f1c40f", "#9b59b6", "#1abc9c", "#e67e22", "#7f8c8d"];

function drawLines(canvas, series, { ymin = null, ymax = null } = {}) {
  const dpr = devicePixelRatio || 1, W = canvas.clientWidth || 300, H = +canvas.getAttribute("height");
  canvas.width = W * dpr; canvas.height = H * dpr;
  const g = canvas.getContext("2d"); g.scale(dpr, dpr); g.clearRect(0, 0, W, H);
  const css = getComputedStyle(document.documentElement), line = css.getPropertyValue("--line"), mu = css.getPropertyValue("--muted");
  const all = series.flatMap(s => s.values).filter(v => v != null && isFinite(v));
  if (!all.length) return;
  let lo = ymin ?? Math.min(...all), hi = ymax ?? Math.max(...all); if (hi - lo < 1e-9) hi = lo + 1;
  const n = Math.max(...series.map(s => s.values.length)), L = 36, B = 18, T0 = 8;
  const X = i => L + (W - L - 8) * (n <= 1 ? .5 : i / (n - 1)), Y = v => T0 + (H - T0 - B) * (1 - (v - lo) / (hi - lo));
  g.strokeStyle = line; g.fillStyle = mu; g.font = "10.5px system-ui"; g.lineWidth = 1;
  for (let i = 0; i <= 4; i++) { const v = lo + (hi - lo) * i / 4, y = Y(v); g.beginPath(); g.moveTo(L, y); g.lineTo(W - 4, y); g.stroke(); g.fillText(v.toFixed(2), 2, y + 3); }
  g.fillText("1", L, H - 4); g.fillText(String(n), W - 18, H - 4);
  series.forEach(s => { g.strokeStyle = s.color; g.lineWidth = 2; g.setLineDash(s.dash || []); g.beginPath();
    s.values.forEach((v, i) => i ? g.lineTo(X(i), Y(v)) : g.moveTo(X(i), Y(v))); g.stroke(); g.setLineDash([]); });
}

const FIELDS = [
  { k: "epochs", l: "Époques", t: "number", min: 1, max: 300 }, { k: "batch_size", l: "Batch", t: "number", min: 1, max: 64 },
  { k: "val_split", l: "Validation (fraction)", t: "number", min: .05, max: .4, step: .05 },
  { k: "size", l: "Taille des images", t: "select", o: [64, 128, 256] },
  { k: "loss", l: "Fonction de perte", t: "select", o: ["bce", "dice", "bce_dice"] },
  { k: "lr", l: "Learning rate (Adam)", t: "number", min: 1e-5, max: .1, step: .0001 },
  { k: "norm", l: "Normalisation", t: "select", o: ["tp4_l2", "unit"] },
  { k: "dropout_scale", l: "Facteur de dropout", t: "number", min: 0, max: 2, step: .25 },
  { k: "early_stopping", l: "Arrêt anticipé (EarlyStopping)", t: "bool", full: true },
  { k: "patience", l: "Patience (époques)", t: "number", min: 1, max: 50 },
  { k: "monitor", l: "Critère surveillé", t: "select", o: ["val_loss", "val_dice"] },
  { k: "augment", l: "Augmentation : miroir horizontal", t: "bool", full: true },
  { k: "name", l: "Nom du modèle (optionnel)", t: "text", full: true },
];
function buildForm(def) {
  T.cfg = { ...def };
  $("#tr-form").replaceChildren(...FIELDS.map(f => {
    const id = "tf_" + f.k, v = T.cfg[f.k];
    let inp;
    if (f.t === "select") { inp = el("select", { id }, f.o.map(o => el("option", { value: o, selected: String(o) === String(v) }, o))); inp.addEventListener("change", () => T.cfg[f.k] = isNaN(+inp.value) ? inp.value : +inp.value); }
    else if (f.t === "bool") { inp = el("input", { type: "checkbox", id, checked: !!v }); inp.addEventListener("change", () => T.cfg[f.k] = inp.checked); }
    else { inp = el("input", { type: f.t, id, value: v, min: f.min, max: f.max, step: f.step }); inp.addEventListener("input", () => T.cfg[f.k] = f.t === "number" ? +inp.value : inp.value); }
    return el("div", { class: "field" + (f.t === "bool" ? " chk" : "") + (f.full ? " full" : "") }, f.t === "bool" ? [inp, el("label", { for: id }, f.l)] : [el("label", { for: id }, f.l), inp]);
  }));
}

async function loadDataset(fd) {
  $("#tr-ds").replaceChildren(el("span", { class: "muted" }, "Chargement…"));
  try {
    const r = await api("/api/train/dataset", { method: "POST", body: fd });
    T.ds = r.dataset_id; $("#tr-start").disabled = false;
    const chips = [chip("Paires image/masque", r.count), chip("Pixels de la région d'intérêt", r.mask_pct + " %"), chip("Masques vides", r.empty_masks)];
    $("#tr-ds").replaceChildren(...chips, ...(r.mask_pct < 20 ? [el("p", { class: "notice", style: "flex-basis:100%" },
      `Classes très déséquilibrées (${100 - r.mask_pct} % de fond) : l'accuracy sera trompeuse. Surveillez le Dice/IoU et essayez la perte « bce_dice ».`)] : []));
  } catch (e) { T.ds = null; $("#tr-start").disabled = true; $("#tr-ds").replaceChildren(el("p", { class: "notice warn", style: "flex-basis:100%" }, e.message)); }
}

function renderJob(s) {
  $("#tr-live").hidden = false;
  $("#tr-bar").style.width = (100 * s.epoch / s.epochs) + "%";
  $("#tr-msg").textContent = `${s.message} — époque ${s.epoch}/${s.epochs} · ${s.elapsed} s` + (s.error ? ` — ${s.error}` : "");
  const h = s.history;
  drawLines($("#ch-loss"), [{ values: h.loss, color: COLORS[1] }, { values: h.val_loss, color: COLORS[0] }]);
  drawLines($("#ch-acc"), [{ values: h.accuracy, color: COLORS[1] }, { values: h.val_accuracy, color: COLORS[0] }], { ymin: 0, ymax: 1 });
  drawLines($("#ch-dice"), [{ values: h.val_dice, color: COLORS[2] }, { values: h.val_iou, color: COLORS[3], dash: [5, 3] }], { ymin: 0, ymax: 1 });
  $("#tr-legend").innerHTML = '<i style="background:#3498db"></i>apprentissage<i style="background:#e74c3c"></i>validation<i style="background:#2ecc71"></i>Dice<i style="background:#f1c40f"></i>IoU';
  const m = s.metrics;
  if (m) {
    $("#tr-metrics").replaceChildren(chip("Accuracy val.", (100 * m.val_accuracy).toFixed(2) + " %"), chip("Loss val.", m.val_loss.toFixed(4)),
      chip("IoU", m.val_iou.toFixed(4)), chip("Dice moyen", m.val_dice.toFixed(4)), chip("Époques", m.epochs_run), chip("Meilleure (val_loss)", m.best_epoch));
    const n = $("#tr-insight"); n.hidden = true;
    if (m.val_accuracy > .85 && m.val_dice < .6) { n.hidden = false; n.textContent = "Accuracy élevée mais Dice/IoU faibles : l'accuracy est dominée par l'arrière-plan (constat du TP4). Fiez-vous au Dice/IoU."; }
    else if (!s.cfg.early_stopping && s.history.val_loss.length > 20 && m.best_epoch < s.history.val_loss.length - 10) { n.hidden = false; n.textContent = `La meilleure perte de validation date de l'époque ${m.best_epoch} sur ${s.history.val_loss.length} : signe de sur-apprentissage. Activez l'arrêt anticipé.`; }
  }
  if (s.samples) { $("#tr-samples").src = s.samples; $("#tr-samples-box").hidden = false; }
}

async function poll() {
  try {
    const s = await api("/api/train/status/" + T.job); renderJob(s);
    if (s.status === "running") { T.timer = setTimeout(poll, 1500); return; }
    $("#tr-start").disabled = !T.ds; $("#tr-stop").disabled = true;
    if (s.status === "error") { setStatus("Échec de l'entraînement", true); return; }
    setStatus("Entraînement terminé");
    S.schema = await api("/api/schema"); renderControls();   // le nouveau modèle apparaît dans « Apprentissage profond »
    refreshRuns(s.model_id && s.model_id.slice(5).replace(/\.keras$/, ""));
  } catch (e) { setStatus(e.message, true); }
}

async function refreshRuns(select) {
  const info = await api("/api/train/info"); T.runs = info.runs;
  if (select) T.sel.add(select);
  renderRuns();
}
function renderRuns() {
  const box = $("#tr-runs");
  if (!T.runs.length) { box.replaceChildren(el("p", { class: "muted" }, "Aucun entraînement enregistré pour l'instant.")); return; }
  const rows = T.runs.map((r, i) => {
    const cb = el("input", { type: "checkbox", checked: T.sel.has(r.id) });
    cb.addEventListener("change", () => { cb.checked ? T.sel.add(r.id) : T.sel.delete(r.id); drawRuns(); });
    const c = r.cfg, m = r.metrics;
    return el("tr", {}, el("td", {}, el("label", {}, cb, el("i", { style: `width:10px;height:10px;border-radius:50%;background:${COLORS[i % 8]};display:inline-block` }), r.name)),
      ...[m.epochs_run + "/" + c.epochs, c.loss, c.early_stopping ? "oui" : "non", (100 * m.val_accuracy).toFixed(1) + " %", m.val_iou.toFixed(3), m.val_dice.toFixed(3)].map(x => el("td", {}, String(x))));
  });
  box.replaceChildren(el("div", { class: "scroll" }, el("table", {}, el("thead", {}, el("tr", {}, ["Modèle", "Époques", "Perte", "Arrêt ant.", "Accuracy", "IoU", "Dice"].map(h => el("th", {}, h)))), el("tbody", {}, rows))));
  const seg = $("#tr-metric-seg"); seg.replaceChildren(...[["val_dice", "Dice"], ["val_iou", "IoU"], ["val_loss", "Loss val."], ["val_accuracy", "Accuracy val."], ["loss", "Loss train"]].map(([k, l]) => {
    const b = el("button", { "aria-pressed": String(T.metric === k) }, l); b.addEventListener("click", () => { T.metric = k; renderRuns(); }); return b; }));
  drawRuns();
}
function drawRuns() {
  const sel = T.runs.map((r, i) => ({ r, i })).filter(x => T.sel.has(x.r.id));
  drawLines($("#ch-runs"), sel.map(x => ({ values: x.r.history[T.metric] || [], color: COLORS[x.i % 8] })), T.metric.match(/dice|iou|accuracy/) ? { ymin: 0, ymax: 1 } : {});
  $("#tr-runs-legend").innerHTML = sel.map(x => `<i style="background:${COLORS[x.i % 8]}"></i>${x.r.name}`).join("");
}

async function initTrain() {
  const info = await api("/api/train/info");
  buildForm(info.defaults); T.runs = info.runs; renderRuns();
  if (!info.available) { const w = $("#tr-warn"); w.hidden = false; w.className = "notice warn"; w.textContent = info.reason + " (puis relancez l'application). L'import de jeu de données reste possible."; }
  $("#tr-zip").addEventListener("change", e => { const f = e.target.files[0]; if (!f) return; $("#tr-zip-label").textContent = f.name; const fd = new FormData(); fd.append("zip", f); loadDataset(fd); });
  $("#tr-load").addEventListener("click", () => { const p = $("#tr-path").value.trim(); if (!p) return; const fd = new FormData(); fd.append("path", p); loadDataset(fd); });
  $("#tr-start").addEventListener("click", async () => {
    try {
      const r = await post("/api/train/start", { dataset_id: T.ds, cfg: T.cfg }); T.job = r.job_id;
      $("#tr-start").disabled = true; $("#tr-stop").disabled = false; $("#tr-samples-box").hidden = true; $("#tr-metrics").replaceChildren(); $("#tr-insight").hidden = true;
      setStatus("Entraînement en cours…"); poll();
    } catch (e) { setStatus(e.message, true); $("#tr-ds").append(el("p", { class: "notice warn", style: "flex-basis:100%" }, e.message)); }
  });
  $("#tr-stop").addEventListener("click", () => T.job && post("/api/train/stop/" + T.job, {}));
}
initTrain().catch(e => setStatus(e.message, true));
