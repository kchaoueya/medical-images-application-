"use strict";
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const el = (tag, attrs = {}, ...kids) => {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") e.className = v;
    else if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
    else if (v !== false && v != null) e.setAttribute(k, v === true ? "" : v);
  }
  kids.flat().forEach(c => e.append(c?.nodeType ? c : document.createTextNode(c ?? "")));
  return e;
};

const S = {
  schema: null, imageId: null, w: 0, h: 0, pipeline: [], methods: {}, seed: { cx: .5, cy: .5, r: .15 },
  polarity: "dark", post: { exclude_bg: false, bg_thr: 15, open_k: 3, largest: true, fill_holes: true, min_area: 50, apply_to_dl: false },
  results: [], pairwise: null, hasGt: false, cmpSel: new Set(), bar: "dice", mx: "dice", busy: false,
};
const NAMES = {};

/* ------------------------------------------------------------------ réseau */
function setStatus(msg, err = false) { const s = $("#status"); s.textContent = msg; s.classList.toggle("err", err); }
async function api(url, opts) {
  const r = await fetch(url, opts);
  const j = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(j.error || `Erreur ${r.status}`);
  return j;
}
const post = (url, body) => api(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
const debounce = (f, ms) => { let t; return (...a) => { clearTimeout(t); t = setTimeout(() => f(...a), ms); }; };

/* ------------------------------------------------------ contrôles de paramètres */
function control(param, value, onChange) {
  const id = "c" + Math.random().toString(36).slice(2, 8);
  if (param.type === "select") {
    const sel = el("select", { id }, param.options.map(o => el("option", { value: o, selected: o === value }, o)));
    sel.addEventListener("change", () => onChange(sel.value));
    return el("div", { class: "ctl" }, el("label", { for: id }, param.label), sel);
  }
  if (param.type === "bool") {
    const cb = el("input", { type: "checkbox", id, checked: !!value });
    cb.addEventListener("change", () => onChange(cb.checked));
    return el("div", { class: "ctl chk" }, el("label", { for: id }, param.label), cb);
  }
  const out = el("output", { for: id }, String(value));
  const r = el("input", { type: "range", id, min: param.min, max: param.max, step: param.step, value });
  r.addEventListener("input", () => { out.textContent = r.value; });
  r.addEventListener("change", () => onChange(param.type === "int" ? parseInt(r.value) : parseFloat(r.value)));
  return el("div", { class: "ctl" }, el("label", { for: id }, param.label), r, out);
}
const defaults = spec => Object.fromEntries(spec.params.map(p => [p.key, p.default]));

/* --------------------------------------------------------------- prétraitement */
function renderPipeline() {
  const ol = $("#pipeline"); ol.replaceChildren();
  $("#pipeline-empty").hidden = S.pipeline.length > 0;
  S.pipeline.forEach((st, i) => {
    const spec = S.schema.preprocess.find(s => s.id === st.op);
    const move = (d) => { const j = i + d; if (j < 0 || j >= S.pipeline.length) return; [S.pipeline[i], S.pipeline[j]] = [S.pipeline[j], S.pipeline[i]]; renderPipeline(); refreshPre(); };
    ol.append(el("li", { class: "step" },
      el("div", { class: "step-h" }, el("span", {}, `${i + 1}. ${spec.name}`),
        el("button", { title: "Monter", "aria-label": "Monter", onclick: () => move(-1) }, "↑"),
        el("button", { title: "Descendre", "aria-label": "Descendre", onclick: () => move(1) }, "↓"),
        el("button", { title: "Supprimer", "aria-label": "Supprimer", onclick: () => { S.pipeline.splice(i, 1); renderPipeline(); refreshPre(); } }, "✕")),
      spec.params.map(p => control(p, st.params[p.key], v => { st.params[p.key] = v; refreshPre(); }))));
  });
}
async function refreshPre() {
  if (!S.imageId) return;
  try {
    const r = await post("/api/preprocess", { image_id: S.imageId, pipeline: S.pipeline });
    $("#img-pre").src = r.image;
    const q = r.quality;
    $("#quality").replaceChildren(
      chip("PSNR vs originale", `${q.psnr.toFixed(1)} dB`), chip("SSIM vs originale", q.ssim.toFixed(3)),
      chip("Moyenne", q.mean.toFixed(1)), chip("Écart-type", q.std.toFixed(1)));
    drawHist(r.hist_before, r.hist_after);
  } catch (e) { setStatus(e.message, true); }
}
const chip = (k, v) => el("span", { class: "chip" }, k, el("b", {}, v));

function drawHist(a, b) {
  const c = $("#hist"), dpr = devicePixelRatio || 1, W = c.clientWidth, H = 170;
  c.width = W * dpr; c.height = H * dpr;
  const g = c.getContext("2d"); g.scale(dpr, dpr); g.clearRect(0, 0, W, H);
  const css = getComputedStyle(document.documentElement);
  const f = x => Math.log1p(x), max = Math.max(...a.map(f), ...b.map(f), 1);
  g.strokeStyle = css.getPropertyValue("--line"); g.lineWidth = 1;
  for (let i = 0; i <= 4; i++) { const y = 10 + (H - 30) * i / 4; g.beginPath(); g.moveTo(30, y); g.lineTo(W, y); g.stroke(); }
  g.fillStyle = css.getPropertyValue("--muted"); g.font = "11px system-ui";
  [0, 64, 128, 192, 255].forEach(v => g.fillText(v, 30 + (W - 34) * v / 255 - 6, H - 4));
  const draw = (h, col, fill) => {
    g.beginPath(); g.moveTo(30, H - 20);
    h.forEach((v, i) => g.lineTo(30 + (W - 34) * i / 255, H - 20 - (H - 30) * f(v) / max));
    g.lineTo(W - 4, H - 20); g.closePath(); g.fillStyle = fill; g.fill(); g.strokeStyle = col; g.lineWidth = 1.5; g.stroke();
  };
  const mu = css.getPropertyValue("--muted").trim(), ac = css.getPropertyValue("--accent").trim();
  draw(a, mu, "rgba(128,128,128,.15)"); draw(b, ac, "rgba(45,212,191,.18)");
}

/* ------------------------------------------------------------------- germe */
function drawSeed() {
  const svg = $("#seed-svg"); svg.setAttribute("viewBox", `0 0 ${S.w} ${S.h}`);
  const cx = S.seed.cx * (S.w - 1), cy = S.seed.cy * (S.h - 1), r = S.seed.r * Math.min(S.w, S.h);
  Object.entries({ cx, cy, r }).forEach(([k, v]) => $("#seed-c").setAttribute(k, v));
  $("#seed-dot").setAttribute("cx", cx); $("#seed-dot").setAttribute("cy", cy); $("#seed-dot").setAttribute("r", Math.max(3, S.w / 150));
}
function initSeedDrag() {
  const svg = $("#seed-svg"); let down = false;
  const move = ev => {
    const b = svg.getBoundingClientRect();
    S.seed.cx = Math.min(Math.max((ev.clientX - b.left) / b.width, 0), 1);
    S.seed.cy = Math.min(Math.max((ev.clientY - b.top) / b.height, 0), 1);
    drawSeed();
  };
  svg.addEventListener("pointerdown", ev => { down = true; svg.setPointerCapture(ev.pointerId); move(ev); });
  svg.addEventListener("pointermove", ev => down && move(ev));
  svg.addEventListener("pointerup", () => { down = false; });
}

/* ---------------------------------------------------- panneau de segmentation */
function renderControls() {
  const sc = $("#seed-controls"); sc.replaceChildren(
    control({ type: "float", label: "Rayon (fraction)", min: .03, max: .5, step: .01 }, S.seed.r, v => { S.seed.r = v; drawSeed(); }));
  const P = [
    { key: "exclude_bg", type: "bool", label: "Exclure le fond noir (connecté aux bords)" },
    { key: "bg_thr", type: "int", label: "Seuil du fond", min: 0, max: 100, step: 1 },
    { key: "open_k", type: "int", label: "Ouverture morphologique", min: 1, max: 15, step: 2 },
    { key: "min_area", type: "int", label: "Surface minimale (px)", min: 0, max: 2000, step: 10 },
    { key: "largest", type: "bool", label: "Garder la plus grande composante" },
    { key: "fill_holes", type: "bool", label: "Remplir les trous" },
    { key: "apply_to_dl", type: "bool", label: "Appliquer aussi aux modèles DL" }];
  $("#post-controls").replaceChildren(...P.map(p => control(p, S.post[p.key], v => { S.post[p.key] = v; })));

  const box = $("#methods"); box.replaceChildren();
  let grp = null;
  for (const m of S.schema.segmentation) {
    if (m.group !== grp) { grp = m.group; box.append(el("div", { class: "group" }, grp)); }
    box.append(methodCard(m.id, m.name, m.params, !!m.needs_seed));
  }
  const dbox = $("#dl-methods"); dbox.replaceChildren();
  S.schema.models.forEach(m => {
    const params = [{ key: "threshold", label: "Seuil de probabilité", type: "float", min: .1, max: .9, step: .05, default: .5 }];
    dbox.append(methodCard(m.id, m.name, params, m.needs_seed, m));
  });
  if (!S.schema.models.some(m => m.kind !== "medsam"))
    dbox.append(el("p", { class: "muted" }, "Aucun modèle dans models/. Déposez-y un U-Net exporté (.onnx, .keras, .h5/.hdf5) — voir le README."));
}
function methodCard(id, name, params, needsSeed, dlMeta) {
  NAMES[id] = name;
  const st = S.methods[id] ??= { on: false, params: Object.fromEntries(params.map(p => [p.key, p.default])) };
  const unavailable = dlMeta && !dlMeta.available;
  const cb = el("input", { type: "checkbox", checked: st.on, disabled: unavailable });
  cb.addEventListener("change", () => { st.on = cb.checked; updateRunBtn(); });
  const pbox = el("div", { class: "params", hidden: true }, params.map(p => control(p, st.params[p.key], v => { st.params[p.key] = v; })));
  if (dlMeta) pbox.prepend(el("p", { class: "muted" }, dlMeta.available ? dlMeta.description : `Indisponible — ${dlMeta.reason}`));
  const gear = el("button", { class: "gear", title: "Paramètres", "aria-label": `Paramètres de ${name}`, "aria-expanded": "false" }, "⚙");
  gear.addEventListener("click", e => { e.preventDefault(); pbox.hidden = !pbox.hidden; gear.setAttribute("aria-expanded", String(!pbox.hidden)); });
  return el("div", { class: "method" },
    el("label", {}, cb, el("span", { class: "grow" }, name),
      needsSeed ? el("span", { class: "badge", title: "Utilise le germe placé sur l'image" }, "germe") : "",
      dlMeta && unavailable ? el("span", { class: "badge warn" }, "indisp.") : "", gear),
    pbox);
}
function updateRunBtn() {
  const any = Object.values(S.methods).some(m => m.on);
  $("#btn-run").disabled = !S.imageId || !any || S.busy;
}

/* ------------------------------------------------------------ segmentation */
async function run() {
  const methods = Object.entries(S.methods).filter(([, m]) => m.on).map(([id, m]) => ({ id, params: m.params }));
  S.busy = true; updateRunBtn(); setStatus(`Calcul de ${methods.length} méthode(s)…`);
  try {
    const r = await post("/api/segment", { image_id: S.imageId, pipeline: S.pipeline, polarity: S.polarity, post: S.post, seed: S.seed, methods });
    S.results = r.results; S.pairwise = r.pairwise; S.hasGt = r.has_gt;
    S.cmpSel = new Set(r.results.filter(x => !x.error).map(x => x.id));
    renderResults(); renderCompare();
    $("#btn-export").removeAttribute("aria-disabled"); $("#btn-export").href = `/api/export/${S.imageId}`;
    const bad = r.results.filter(x => x.error).length;
    setStatus(bad ? `Terminé — ${bad} méthode(s) en erreur` : "Terminé", !!bad);
    showTab("t-results");
  } catch (e) { setStatus(e.message, true); }
  S.busy = false; updateRunBtn();
}
const fmt = (v, d = 3) => v == null ? "—" : Number(v).toFixed(d);

function renderResults() {
  const g = $("#results-grid"); g.replaceChildren();
  S.results.forEach(r => {
    if (r.error) { g.append(el("div", { class: "res" }, el("div", { class: "res-b" }, el("div", { class: "res-h" }, el("b", {}, r.name)), el("div", { class: "err-box" }, r.error)))); return; }
    const img = el("img", { src: r.overlay, alt: `Segmentation ${r.name}` });
    let showMask = false;
    const box = el("div", { class: "imgbox", title: "Cliquer pour basculer superposition / masque", onclick: () => { showMask = !showMask; img.src = showMask ? r.mask : r.overlay; } }, img);
    const kv = [["Temps", `${r.time_ms} ms`], ["Aire", `${r.stats.area_pct} % (${r.stats.area_px} px)`], ["Périmètre", r.stats.perimeter], ["Circularité", r.stats.circularity]];
    if (r.vs_gt) kv.push(["Dice / IoU", `${fmt(r.vs_gt.dice)} / ${fmt(r.vs_gt.iou)}`], ["Hausdorff", r.vs_gt.hausdorff == null ? "—" : `${r.vs_gt.hausdorff.toFixed(1)} px`]);
    Object.entries(r.info || {}).forEach(([k, v]) => kv.push([k, Array.isArray(v) ? v.join(", ") : v]));
    g.append(el("div", { class: "res" }, box, el("div", { class: "res-b" },
      el("div", { class: "res-h" }, el("b", {}, el("i", { style: `background:${r.color}` }), r.name), el("span", { class: "badge" }, r.family === "dl" ? "Deep Learning" : "Conventionnelle")),
      el("div", { class: "kv" }, kv.flatMap(([k, v]) => [el("span", {}, k), el("span", {}, String(v))])))));
  });
}

/* -------------------------------------------------------------- comparaison */
function renderCompare() {
  const ok = S.results.filter(r => !r.error);
  $("#cmp-empty").hidden = ok.length >= 2; $("#cmp").hidden = ok.length < 2;
  if (ok.length < 2) return;
  const chips = $("#cmp-chips"); chips.replaceChildren(...ok.map(r => {
    const c = el("span", { class: "chip toggle", role: "button", tabindex: "0", "aria-pressed": String(S.cmpSel.has(r.id)) }, el("i", { style: `background:${r.color}` }), r.name);
    const t = () => { S.cmpSel.has(r.id) ? S.cmpSel.delete(r.id) : S.cmpSel.add(r.id); c.setAttribute("aria-pressed", String(S.cmpSel.has(r.id))); updateOverlay(); };
    c.addEventListener("click", t); c.addEventListener("keydown", e => (e.key === "Enter" || e.key === " ") && (e.preventDefault(), t()));
    return c;
  }));
  const sm = $("#slider-method"); sm.replaceChildren(...ok.map(r => el("option", { value: r.id }, r.name)));
  updateOverlay(); updateSlider();
  // tableau
  const head = ["Méthode", "Temps (ms)", "Aire (%)", "Périm.", "Circ."].concat(S.hasGt ? ["Dice", "IoU", "Hausdorff"] : []);
  const rows = ok.map(r => [r.name, r.time_ms, r.stats.area_pct, r.stats.perimeter, r.stats.circularity].concat(S.hasGt ? [fmt(r.vs_gt?.dice), fmt(r.vs_gt?.iou), r.vs_gt?.hausdorff == null ? "—" : r.vs_gt.hausdorff.toFixed(1)] : []));
  const best = S.hasGt ? Math.max(...ok.map(r => r.vs_gt?.dice ?? -1)) : null;
  $("#cmp-table").replaceChildren(el("thead", {}, el("tr", {}, head.map(h => el("th", {}, h)))),
    el("tbody", {}, rows.map((row, i) => el("tr", {}, row.map((c, j) => el("td", j === 5 && ok[i].vs_gt?.dice === best ? { style: "color:var(--accent);font-weight:700" } : {}, String(c)))))));
  renderMatrix(); renderBars();
}
async function updateOverlay() {
  try { const r = await post("/api/compare", { image_id: S.imageId, methods: [...S.cmpSel], show_gt: $("#show-gt").checked }); $("#cmp-img").src = r.image; }
  catch (e) { setStatus(e.message, true); }
}
function updateSlider() {
  const r = S.results.find(x => x.id === $("#slider-method").value); if (!r) return;
  $("#sl-a").src = $("#img-pre").src; $("#sl-b").src = r.overlay; moveSlider($("#sl-range").value);
}
function moveSlider(v) { $("#sl-b").style.clipPath = `inset(0 ${100 - v}% 0 0)`; $("#sl-h").style.left = v + "%"; }
function renderMatrix() {
  const p = S.pairwise; if (!p) return;
  const M = p[S.mx], names = p.names.map(id => NAMES[id] || id);
  $("#mx-kind").textContent = S.mx === "dice" ? "(Dice)" : "(IoU)";
  $("#matrix").replaceChildren(el("thead", {}, el("tr", {}, el("th", {}, ""), names.map(n => el("th", { title: n }, n)))),
    el("tbody", {}, M.map((row, i) => el("tr", {}, el("th", { title: names[i] }, names[i]),
      row.map(v => el("td", { style: `background:color-mix(in srgb,var(--accent) ${Math.round(v * 85)}%,transparent);color:${v > .6 ? "var(--accent-ink)" : "var(--text)"}` }, v.toFixed(2)))))));
}
function renderBars() {
  const ok = S.results.filter(r => !r.error);
  const val = r => ({ dice: r.vs_gt?.dice, iou: r.vs_gt?.iou, time: r.time_ms, area: r.stats.area_pct })[S.bar];
  const vals = ok.map(val).filter(v => v != null);
  if (!vals.length) { $("#bars").replaceChildren(el("p", { class: "muted" }, "Importez un masque de référence pour comparer au Dice / IoU.")); return; }
  const max = S.bar === "dice" || S.bar === "iou" ? 1 : Math.max(...vals, 1e-9);
  $("#bars").replaceChildren(...ok.map(r => { const v = val(r); return el("div", { class: "bar" }, el("span", { title: r.name }, r.name),
    el("div", { class: "t" }, el("i", { style: `width:${v == null ? 0 : v / max * 100}%;background:${r.color}` })),
    el("span", {}, v == null ? "—" : (S.bar === "dice" || S.bar === "iou" ? v.toFixed(3) : v.toFixed(1)))); }));
}

/* ----------------------------------------------------------------- image */
async function loadImage(file, maskFile) {
  setStatus("Import…");
  const fd = new FormData(); fd.append("image", file); if (maskFile) fd.append("mask", maskFile);
  try {
    const r = await api("/api/upload", { method: "POST", body: fd });
    Object.assign(S, { imageId: r.image_id, w: r.width, h: r.height, results: [], pairwise: null, hasGt: r.has_gt });
    $("#img-orig").src = r.image; $("#empty").hidden = true; $("#image-view").hidden = false;
    $("#img-info").textContent = `${r.name} — ${r.width}×${r.height} px${r.has_gt ? " · masque de référence chargé" : ""}`;
    $("#mask-label").textContent = r.has_gt ? "Masque de référence chargé ✓" : "Masque de référence (optionnel)";
    $("#results-grid").replaceChildren(el("p", { class: "muted" }, "Lancez une segmentation pour voir les résultats."));
    $("#cmp").hidden = true; $("#cmp-empty").hidden = false;
    $("#btn-export").setAttribute("aria-disabled", "true");
    drawSeed(); await refreshPre(); updateRunBtn(); setStatus("Image prête"); showTab("t-image");
  } catch (e) { setStatus(e.message, true); }
}
async function loadSample(s) {
  const [im, mk] = await Promise.all([fetch(s.image).then(r => r.blob()), fetch(s.mask).then(r => r.blob())]);
  const polarity = s.name.startsWith("irm") ? "bright" : "dark";
  $("#polarity").value = S.polarity = polarity;
  S.post.exclude_bg = s.name !== "echo_lesion"; renderControls();
  await loadImage(new File([im], s.name + ".png", { type: "image/png" }), new File([mk], s.name + "_mask.png", { type: "image/png" }));
}

function showTab(id) {
  $$(".tabpanel").forEach(p => p.hidden = p.id !== id);
  $$(".tabs button").forEach(b => b.setAttribute("aria-selected", String(b.dataset.tab === id)));
}

/* ------------------------------------------------------------------ init */
async function init() {
  S.schema = await api("/api/schema");
  S.schema.preprocess.forEach(p => NAMES[p.id] = p.name);
  const sel = $("#op-select"); let grp = null, og = null;
  S.schema.preprocess.forEach(p => { if (p.group !== grp) { grp = p.group; og = el("optgroup", { label: grp }); sel.append(og); } og.append(el("option", { value: p.id }, p.name)); });
  $("#op-add").addEventListener("click", () => {
    const spec = S.schema.preprocess.find(s => s.id === sel.value);
    S.pipeline.push({ op: spec.id, params: defaults(spec) }); renderPipeline(); refreshPre();
  });
  renderPipeline(); renderControls(); initSeedDrag();
  $("#polarity").addEventListener("change", e => S.polarity = e.target.value);
  $("#f-image").addEventListener("change", e => e.target.files[0] && loadImage(e.target.files[0], $("#f-mask").files[0]));
  $("#f-mask").addEventListener("change", async e => {
    const f = e.target.files[0]; if (!f || !S.imageId) { if (f) setStatus("Importez d'abord l'image", true); return; }
    const fd = new FormData(); fd.append("mask", f);
    try { await api(`/api/mask/${S.imageId}`, { method: "POST", body: fd }); S.hasGt = true; $("#mask-label").textContent = "Masque de référence chargé ✓"; setStatus("Masque chargé — relancez la segmentation"); }
    catch (err) { setStatus(err.message, true); }
  });
  try {
    const sm = await api("/api/samples");
    $("#samples").replaceChildren(...sm.samples.map(s => el("button", { class: "samples", onclick: () => loadSample(s) }, s.name.replace("_", " "))));
  } catch { /* exemples facultatifs */ }
  $$(".tabs button").forEach(b => b.addEventListener("click", () => showTab(b.dataset.tab)));
  $("#btn-run").addEventListener("click", run);
  $("#show-gt").addEventListener("change", updateOverlay);
  $("#slider-method").addEventListener("change", updateSlider);
  $("#sl-range").addEventListener("input", e => moveSlider(e.target.value));
  const mx = k => { S.mx = k; $("#mx-dice").setAttribute("aria-pressed", String(k === "dice")); $("#mx-iou").setAttribute("aria-pressed", String(k === "iou")); renderMatrix(); };
  $("#mx-dice").addEventListener("click", () => mx("dice")); $("#mx-iou").addEventListener("click", () => mx("iou"));
  $$("[data-bar]").forEach(b => b.addEventListener("click", () => { S.bar = b.dataset.bar; $$("[data-bar]").forEach(x => x.setAttribute("aria-pressed", String(x === b))); renderBars(); }));
  $("#img-pre").addEventListener("load", () => { if ($("#sl-a").src) updateSlider(); });
  addEventListener("resize", debounce(() => S.imageId && refreshPre(), 200));
}
init().catch(e => setStatus(e.message, true));
