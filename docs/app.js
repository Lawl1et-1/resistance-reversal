/* Phase plate — the figures draw themselves from data.json, and the phase portrait
   integrates the same update rule the experiments use. Nothing here is a picture of a
   result; it is the result, plotted. */

const NS = 'http://www.w3.org/2000/svg';
const INK = '#0d1116';
const ANNOT = '#4e5862';
const RES = '#c3351a';
const SEN = '#17407e';
const RES_FIELD = '#e3a699';
const SEN_FIELD = '#a9b9d4';
const RULE = 'rgba(13,17,22,0.26)';

/* --- language ------------------------------------------------------------------
   English lives in index.html; French comes from i18n.fr.js keyed by data-i18n.
   Switching re-renders everything generated, because the charts, tables and ledger
   build their own text. */
let LANG = (() => {
  try { return localStorage.getItem('lang') || 'en'; } catch { return 'en'; }
})();
const FR = () => (window.FR || { html: {}, ui: {} });
const isFR = () => LANG === 'fr';
const T = () => (isFR() ? FR().ui : null);
const splitLabel = (d, key) => (isFR() && FR().ui.splits[key]) || d.splits[key] || key;
const modelLabel = (m) => (isFR() && FR().ui.models[m.model]) || m.label;

const fmt = (v, d = 3) => {
  if (v === null || v === undefined || Number.isNaN(v)) return isFR() ? 's.o.' : 'n/a';
  const s = v.toFixed(d);
  return isFR() ? s.replace('.', ',') : s;
};
const pct = (v) => {
  if (v === null || v === undefined) return '—';
  const s = (v * 100).toFixed(1);
  return isFR() ? `${s.replace('.', ',')} %` : `${s}%`;
};

/* ---------------------------------------------------------------- phase dynamics ---
   The simulator's toggle switch, reduced to its two master regulators:
     x <- (1-a)x + a*sigmoid(w_self*x - w_cross*y + u)
   Same relaxation constant and gains as configs/full.yaml. A knockdown removes a
   regulator's drive, which is what moves the separatrix rather than the cell. */
const W_SELF = 3.5, W_CROSS = 3.5, ALPHA = 0.35;
const sigmoid = (x) => 1 / (1 + Math.exp(-Math.max(-40, Math.min(40, x))));

const REGULATORS = [
  { id: 'R1', label: 'KD R1', drives: 'res', mag: 1.55 },
  { id: 'R2', label: 'KD R2', drives: 'res', mag: 1.15 },
  { id: 'R3', label: 'KD R3', drives: 'res', mag: 0.72 },
  { id: 'R4', label: 'KD R4', drives: 'sen', mag: 1.1 },
];
const knocked = new Set();

function drive() {
  // Knocking down a resistance driver removes its push on the resistance master and the
  // repression it placed on the sensitive one; a sensitive driver is the mirror image.
  let uRes = 0, uSen = 0;
  for (const r of REGULATORS) {
    if (!knocked.has(r.id)) continue;
    if (r.drives === 'res') { uRes -= r.mag; uSen += 0.5 * r.mag; }
    else { uSen -= r.mag; uRes += 0.5 * r.mag; }
  }
  return { uRes, uSen };
}

function step(res, sen, u) {
  const nRes = (1 - ALPHA) * res + ALPHA * sigmoid(W_SELF * res - W_CROSS * sen + u.uRes);
  const nSen = (1 - ALPHA) * sen + ALPHA * sigmoid(W_SELF * sen - W_CROSS * res + u.uSen);
  return [nRes, nSen];
}

function relax(res, sen, u, steps = 240) {
  const path = [[res, sen]];
  for (let i = 0; i < steps; i++) {
    [res, sen] = step(res, sen, u);
    path.push([res, sen]);
    if (i > 2) {
      const [pr, ps] = path[path.length - 2];
      if (Math.abs(pr - res) < 1e-6 && Math.abs(ps - sen) < 1e-6) break;
    }
  }
  return path;
}

/* Which basin a starting point ends in — this is what paints the fields. */
const basinOf = (res, sen, u) => {
  const p = relax(res, sen, u, 160);
  const [r, s] = p[p.length - 1];
  return s > r ? 'sen' : 'res';
};

function drawPhase(canvas, opts = {}) {
  const ctx = canvas.getContext('2d');
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const cssW = canvas.clientWidth || canvas.width;
  const cssH = opts.heroFill ? (canvas.clientHeight || canvas.height) : cssW * (opts.ratio || 0.52);
  canvas.width = Math.round(cssW * dpr);
  canvas.height = Math.round(cssH * dpr);
  if (!opts.heroFill) canvas.style.height = `${cssH}px`;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, cssW, cssH);

  const pad = opts.heroFill ? 0 : 1;
  const X = (r) => pad + r * (cssW - 2 * pad);
  const Y = (s) => cssH - pad - s * (cssH - 2 * pad);
  const u = drive();

  // Basin fields, at page scale: classify a coarse grid and flood each cell.
  const cols = Math.max(24, Math.round(cssW / 14));
  const rows = Math.max(16, Math.round(cssH / 14));
  for (let i = 0; i < cols; i++) {
    for (let j = 0; j < rows; j++) {
      const r = (i + 0.5) / cols, s = (j + 0.5) / rows;
      ctx.fillStyle = basinOf(r, s, u) === 'sen' ? SEN_FIELD : RES_FIELD;
      ctx.globalAlpha = opts.heroFill ? 0.62 : 0.5;
      ctx.fillRect(X(r) - (cssW / cols) / 2 - 1, Y(s) - (cssH / rows) / 2 - 1,
        cssW / cols + 2, cssH / rows + 2);
    }
  }
  ctx.globalAlpha = 1;

  // Trajectories from a lattice of starts — the plate's ink curves.
  ctx.lineWidth = 1;
  ctx.strokeStyle = opts.heroFill ? 'rgba(13,17,22,0.22)' : 'rgba(13,17,22,0.38)';
  const lat = opts.heroFill ? 6 : 7;
  for (let i = 1; i < lat; i++) {
    for (let j = 1; j < lat; j++) {
      const path = relax(i / lat, j / lat, u, 70);
      ctx.beginPath();
      path.forEach(([r, s], k) => (k ? ctx.lineTo(X(r), Y(s)) : ctx.moveTo(X(r), Y(s))));
      ctx.stroke();
      // arrowhead at the first clear move, so direction reads without animation
      const a = path[Math.min(5, path.length - 1)], b = path[Math.min(12, path.length - 1)];
      if (a && b) {
        const ang = Math.atan2(Y(b[1]) - Y(a[1]), X(b[0]) - X(a[0]));
        if (Number.isFinite(ang) && Math.hypot(X(b[0]) - X(a[0]), Y(b[1]) - Y(a[1])) > 4) {
          ctx.beginPath();
          ctx.moveTo(X(b[0]), Y(b[1]));
          ctx.lineTo(X(b[0]) - 5 * Math.cos(ang - 0.4), Y(b[1]) - 5 * Math.sin(ang - 0.4));
          ctx.lineTo(X(b[0]) - 5 * Math.cos(ang + 0.4), Y(b[1]) - 5 * Math.sin(ang + 0.4));
          ctx.closePath();
          ctx.fillStyle = 'rgba(13,17,22,0.38)';
          ctx.fill();
        }
      }
    }
  }

  // Separatrix: the boundary between the two classified fields, traced by bisection.
  ctx.beginPath();
  let started = false;
  for (let j = 0; j <= rows * 2; j++) {
    const s = j / (rows * 2);
    let lo = 0, hi = 1;
    if (basinOf(lo, s, u) === basinOf(hi, s, u)) continue;
    for (let k = 0; k < 18; k++) {
      const mid = (lo + hi) / 2;
      if (basinOf(mid, s, u) === basinOf(lo, s, u)) lo = mid; else hi = mid;
    }
    const r = (lo + hi) / 2;
    started ? ctx.lineTo(X(r), Y(s)) : (ctx.moveTo(X(r), Y(s)), started = true);
  }
  ctx.strokeStyle = INK;
  ctx.lineWidth = opts.heroFill ? 2 : 1.75;
  ctx.stroke();

  // Fixed points.
  const marks = [
    { p: relax(0.92, 0.06, u).pop(), fill: RES },
    { p: relax(0.06, 0.92, u).pop(), fill: SEN },
  ];
  for (const m of marks) {
    ctx.beginPath();
    ctx.arc(X(m.p[0]), Y(m.p[1]), opts.heroFill ? 7 : 5.5, 0, Math.PI * 2);
    ctx.fillStyle = m.fill;
    ctx.fill();
    ctx.strokeStyle = INK;
    ctx.lineWidth = 1.25;
    ctx.stroke();
  }

  if (opts.cell) {
    const path = relax(opts.cell[0], opts.cell[1], u);
    ctx.beginPath();
    path.forEach(([r, s], k) => (k ? ctx.lineTo(X(r), Y(s)) : ctx.moveTo(X(r), Y(s))));
    ctx.strokeStyle = INK;
    ctx.lineWidth = 2.5;
    ctx.stroke();
    const [er, es] = path[path.length - 1];
    ctx.beginPath();
    ctx.arc(X(er), Y(es), 6, 0, Math.PI * 2);
    ctx.fillStyle = es > er ? SEN : RES;
    ctx.fill();
    ctx.strokeStyle = INK;
    ctx.lineWidth = 1.5;
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(X(opts.cell[0]), Y(opts.cell[1]), 7, 0, Math.PI * 2);
    ctx.strokeStyle = INK;
    ctx.lineWidth = 2;
    ctx.stroke();
  }
}

/* ------------------------------------------------------------------- hero plate --- */
const hero = document.getElementById('phase-hero');
function sizeHero() {
  if (!hero) return;
  hero.style.width = '100%';
  hero.style.height = '100%';
  drawPhase(hero, { heroFill: true });
}

/* ------------------------------------------------------------ interactive plate --- */
const inter = document.getElementById('phase-interactive');
let cell = [0.9, 0.08];

function redrawInteractive() {
  if (inter) drawPhase(inter, { cell, ratio: 0.5 });
}

function pointToState(ev) {
  const rect = inter.getBoundingClientRect();
  const cx = (ev.touches ? ev.touches[0].clientX : ev.clientX) - rect.left;
  const cy = (ev.touches ? ev.touches[0].clientY : ev.clientY) - rect.top;
  return [
    Math.max(0.02, Math.min(0.98, cx / rect.width)),
    Math.max(0.02, Math.min(0.98, 1 - cy / rect.height)),
  ];
}

if (inter) {
  let dragging = false;
  const start = (e) => { dragging = true; cell = pointToState(e); redrawInteractive(); e.preventDefault(); };
  const move = (e) => { if (dragging) { cell = pointToState(e); redrawInteractive(); e.preventDefault(); } };
  const end = () => { dragging = false; };
  inter.addEventListener('pointerdown', start);
  inter.addEventListener('pointermove', move);
  window.addEventListener('pointerup', end);
  inter.addEventListener('keydown', (e) => {
    const d = 0.04;
    const map = { ArrowLeft: [-d, 0], ArrowRight: [d, 0], ArrowUp: [0, d], ArrowDown: [0, -d] };
    if (!map[e.key]) return;
    cell = [
      Math.max(0.02, Math.min(0.98, cell[0] + map[e.key][0])),
      Math.max(0.02, Math.min(0.98, cell[1] + map[e.key][1])),
    ];
    redrawInteractive();
    e.preventDefault();
  });

  const box = document.getElementById('knockdowns');
  REGULATORS.forEach((r) => {
    const b = document.createElement('button');
    b.className = 'knock';
    b.type = 'button';
    b.textContent = isFR() ? FR().ui.kdLabel(r.id) : r.label;
    b.dataset.reg = r.id;
    b.setAttribute('aria-pressed', 'false');
    b.title = isFR()
      ? (r.drives === 'res' ? FR().ui.kdResistance : FR().ui.kdSensitivity)
      : (r.drives === 'res' ? 'Resistance driver' : 'Sensitivity driver');
    b.addEventListener('click', () => {
      knocked.has(r.id) ? knocked.delete(r.id) : knocked.add(r.id);
      b.setAttribute('aria-pressed', String(knocked.has(r.id)));
      redrawInteractive();
      sizeHero();
    });
    box.appendChild(b);
  });
}

/* ------------------------------------------------------------------ svg helpers --- */
function svg(w, h) {
  const s = document.createElementNS(NS, 'svg');
  s.setAttribute('viewBox', `0 0 ${w} ${h}`);
  s.setAttribute('role', 'img');
  return s;
}
function el(tag, attrs, parent) {
  const n = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
  if (parent) parent.appendChild(n);
  return n;
}

/* ------------------------------------------------------------------- the figures --- */
function coverageChart(root, data) {
  const W = 900, H = 330, L = 40, R = 20, T = 30, B = 76;
  const rows = data.coverage;
  const s = svg(W, H);
  s.setAttribute('aria-label',
    'Conformal coverage by split against the 90% nominal target. The unseen-context split, where the assumption is violated, falls far below.');
  const bw = (W - L - R) / rows.length;
  const y = (v) => T + (1 - v) * (H - T - B);

  el('line', { x1: L, y1: y(0), x2: W - R, y2: y(0), stroke: RULE }, s);
  [0.25, 0.5, 0.75, 1].forEach((g) => {
    el('line', { x1: L, y1: y(g), x2: W - R, y2: y(g), stroke: RULE, 'stroke-dasharray': '2 4' }, s);
    const t = el('text', { x: L - 8, y: y(g) + 3, 'text-anchor': 'end' }, s);
    t.textContent = `${g * 100}`;
  });

  rows.forEach((r, i) => {
    const x = L + i * bw + bw * 0.22;
    const w = bw * 0.56;
    const top = y(r.coverage);
    el('rect', {
      x, y: top, width: w, height: y(0) - top,
      fill: r.exchangeable ? SEN_FIELD : RES_FIELD,
      stroke: r.exchangeable ? SEN : RES, 'stroke-width': 1.5,
    }, s);
    if (r.lo != null && r.hi != null) {
      const cx = x + w / 2;
      el('line', { x1: cx, y1: y(r.lo), x2: cx, y2: y(r.hi), stroke: INK, 'stroke-width': 1.5 }, s);
      el('line', { x1: cx - 5, y1: y(r.lo), x2: cx + 5, y2: y(r.lo), stroke: INK, 'stroke-width': 1.5 }, s);
      el('line', { x1: cx - 5, y1: y(r.hi), x2: cx + 5, y2: y(r.hi), stroke: INK, 'stroke-width': 1.5 }, s);
    }
    const v = el('text', { x: x + w / 2, y: top - 9, 'text-anchor': 'middle', class: 'bar-label' }, s);
    v.textContent = pct(r.coverage);

    const covLabel = (isFR() && FR().ui.splits[r.split]) || r.label;
    covLabel.split(/,\s*/).forEach((line, k) => {
      const t = el('text', { x: x + w / 2, y: y(0) + 18 + k * 12, 'text-anchor': 'middle' }, s);
      t.textContent = line;
    });
    if (!r.exchangeable) {
      const t = el('text', { x: x + w / 2, y: y(0) + 18 + 24, 'text-anchor': 'middle', fill: RES }, s);
      t.textContent = isFR() ? FR().ui.assumptionViolated : 'assumption violated';
    }
  });

  const nom = data.verdicts.h2.nominal;
  el('line', { x1: L, y1: y(nom), x2: W - R, y2: y(nom), class: 'nominal' }, s);
  const nt = el('text', { x: W - R, y: y(nom) - 7, 'text-anchor': 'end', fill: RES }, s);
  nt.textContent = isFR()
    ? `${FR().ui.nominal} ${nom * 100} %`
    : `nominal ${nom * 100}%`;
  root.appendChild(s);
}

function accuracyChart(root, data) {
  const splits = Object.keys(data.splits);
  const models = data.accuracy.filter((m) => m.role === 'model');
  const ctrl = data.accuracy.find((m) => m.model === 'shuffled_labels');
  const W = 900, H = 360, L = 40, R = 150, T = 24, B = 70;
  const s = svg(W, H);
  s.setAttribute('aria-label',
    'Prediction accuracy by split for each model, with the shuffled-label control marked as a dashed line in each group.');
  const gw = (W - L - R) / splits.length;
  const y = (v) => T + (1 - Math.max(0, v)) * (H - T - B);

  el('line', { x1: L, y1: y(0), x2: W - R, y2: y(0), stroke: RULE }, s);
  [0.25, 0.5, 0.75, 1].forEach((g) => {
    el('line', { x1: L, y1: y(g), x2: W - R, y2: y(g), stroke: RULE, 'stroke-dasharray': '2 4' }, s);
    const t = el('text', { x: L - 8, y: y(g) + 3, 'text-anchor': 'end' }, s);
    t.textContent = g.toFixed(2);
  });

  splits.forEach((sp, gi) => {
    const x0 = L + gi * gw;
    const bw = (gw * 0.76) / models.length;
    models.forEach((m, mi) => {
      const cellv = m.by_split[sp];
      if (!cellv || cellv.value == null) return;
      const x = x0 + gw * 0.12 + mi * bw;
      const isFlow = m.model === 'pertflow';
      el('rect', {
        x, y: y(cellv.value), width: bw * 0.8, height: y(0) - y(cellv.value),
        fill: isFlow ? RES_FIELD : SEN_FIELD,
        stroke: isFlow ? RES : SEN, 'stroke-width': 1,
      }, s);
      if (cellv.lo != null) {
        const cx = x + bw * 0.4;
        el('line', { x1: cx, y1: y(cellv.lo), x2: cx, y2: y(cellv.hi), stroke: INK, 'stroke-width': 1 }, s);
      }
    });
    const c = ctrl && ctrl.by_split[sp];
    if (c && c.value != null) {
      el('line', {
        x1: x0 + gw * 0.06, y1: y(c.value), x2: x0 + gw * 0.94, y2: y(c.value),
        stroke: INK, 'stroke-width': 1.5, 'stroke-dasharray': '6 3',
      }, s);
    }
    splitLabel(data, sp).split(/,\s*/).forEach((line, k) => {
      const t = el('text', { x: x0 + gw / 2, y: y(0) + 18 + k * 12, 'text-anchor': 'middle' }, s);
      t.textContent = line;
    });
  });

  models.forEach((m, i) => {
    const yy = T + 6 + i * 16;
    const isFlow = m.model === 'pertflow';
    el('rect', {
      x: W - R + 8, y: yy - 8, width: 11, height: 11,
      fill: isFlow ? RES_FIELD : SEN_FIELD, stroke: isFlow ? RES : SEN,
    }, s);
    const t = el('text', { x: W - R + 25, y: yy + 1 }, s);
    t.textContent = modelLabel(m);
  });
  const ly = T + 6 + models.length * 16;
  el('line', {
    x1: W - R + 8, y1: ly - 3, x2: W - R + 19, y2: ly - 3,
    stroke: INK, 'stroke-width': 1.5, 'stroke-dasharray': '6 3',
  }, s);
  const lt = el('text', { x: W - R + 25, y: ly + 1 }, s);
  lt.textContent = isFR() ? FR().ui.shuffledControl : 'shuffled control';
  root.appendChild(s);
}

function accuracyTable(root, data) {
  const splits = Object.keys(data.splits);
  const t = document.createElement('table');
  const cap = isFR() ? FR().ui.accCaption : 'Pearson-delta by split · 5 seeds · hierarchical 95% intervals';
  const colModel = isFR() ? FR().ui.colModel : 'Model';
  t.innerHTML = `<caption>${cap}</caption>
    <thead><tr><th>${colModel}</th>${splits.map((s) => `<th>${splitLabel(data, s)}</th>`).join('')}</tr></thead>`;
  const tb = document.createElement('tbody');
  data.accuracy.forEach((m) => {
    const tr = document.createElement('tr');
    tr.dataset.role = m.role;
    tr.innerHTML = `<td>${modelLabel(m)}</td>` + splits.map((s) => {
      const c = m.by_split[s];
      if (!c || c.value == null) return `<td>${isFR() ? FR().ui.na : 'n/a'}</td>`;
      const lead = (m.model === 'pertflow' && s === 'unseen_context') ? ' class="lead"' : '';
      return `<td${lead}>${fmt(c.value)}</td>`;
    }).join('');
    tb.appendChild(tr);
  });
  t.appendChild(tb);
  root.appendChild(t);
}

function searchTable(root, data) {
  const t = document.createElement('table');
  const u = isFR() ? FR().ui : null;
  t.innerHTML = `<caption>${u ? u.searchCaption : 'Intervention search · greedy · verified against the simulator oracle'}</caption>
    <thead><tr><th>${u ? u.colLambda : 'Uncertainty penalty λ'}</th><th>${u ? u.colVerified : 'Verified reversal'}</th><th>${u ? u.colInterval : '95% interval'}</th><th>${u ? u.colRegret : 'Regret vs best possible'}</th></tr></thead>`;
  const tb = document.createElement('tbody');
  data.search.forEach((r) => {
    const tr = document.createElement('tr');
    tr.innerHTML = `<td>${fmt(r.lam, 1)}</td><td${r.lam === 0 ? ' class="lead"' : ''}>${fmt(r.verified)}</td>
      <td>${fmt(r.lo)} – ${fmt(r.hi)}</td><td>${fmt(r.regret, 4)}</td>`;
    tb.appendChild(tr);
  });
  t.appendChild(tb);
  root.appendChild(t);
}

/* State by stroke pattern, never hue: unbroken = supported, gapped = inconclusive,
   open ring = untestable. Readable without colour. */
function stateMark(kind) {
  const s = svg(54, 14);
  s.setAttribute('width', 54);
  s.setAttribute('height', 14);
  s.setAttribute('aria-hidden', 'true');
  const base = { y1: 7, y2: 7, stroke: INK, 'stroke-width': 2 };
  if (kind === 'supported') {
    el('line', { x1: 0, x2: 54, ...base }, s);
  } else if (kind === 'inconclusive') {
    el('line', { x1: 0, x2: 20, ...base }, s);
    el('line', { x1: 34, x2: 54, ...base }, s);
  } else {
    el('line', { x1: 0, x2: 38, ...base }, s);
    el('circle', { cx: 45, cy: 7, r: 5, fill: 'none', stroke: INK, 'stroke-width': 2 }, s);
  }
  return s;
}

function ledger(root, data) {
  const v = data.verdicts;
  const rows = [
    {
      id: 'H1a', kind: 'inconclusive',
      claim: 'Deep models do not reliably beat strong linear baselines.',
      why: `${v.h1a.descriptive}. The verdict flips depending on how "reliably beats" is defined, and the definition that produced it was chosen after seeing the counts — so it is reported as unresolved.`,
    },
    {
      id: 'H1b', kind: 'supported',
      claim: 'A reversal-aware evaluation reorders the models.',
      why: `The winner changes in ${v.h1b.changes} of ${v.h1b.of} comparisons between the conventional metric and one that measures intervention ranking.`,
    },
    {
      id: 'H2', kind: 'supported',
      claim: 'Conformal calibration reaches its target on unseen perturbations, and abstention lowers error.',
      why: `Holds on all ${v.h2.valid} splits where the exchangeability assumption applies. It fails, as theory predicts, on the one where it does not.`,
    },
    {
      id: 'H3', kind: 'untestable',
      claim: 'Penalising uncertainty improves intervention search.',
      why: `${v.h3.why}. Uncertainty-blind search already reaches ${pct(1 - v.h3.blind_regret)} of the best possible intervention. This is a flaw in the design, not evidence against the idea.`,
    },
    {
      id: 'H4 · H5', kind: 'untestable',
      claim: 'Active selection beats random; top targets are enriched for known biology.',
      why: 'Not run. H5 needs real data, which this project never acquired.',
    },
  ];
  if (isFR()) {
    const L = FR().ui.ledger;
    rows[0].claim = L.h1a.claim; rows[0].why = L.h1a.why(v.h1a);
    rows[1].claim = L.h1b.claim; rows[1].why = L.h1b.why(v.h1b);
    rows[2].claim = L.h2.claim;  rows[2].why = L.h2.why(v.h2);
    rows[3].claim = L.h3.claim;  rows[3].why = L.h3.why(v.h3, pct);
    rows[4].claim = L.h45.claim; rows[4].why = L.h45.why();
  }
  rows.forEach((r) => {
    const dt = document.createElement('dt');
    dt.textContent = r.id;
    const dd = document.createElement('dd');
    dd.innerHTML = `<strong>${r.claim}</strong><p class="why">${r.why}</p>`;
    const st = document.createElement('dd');
    st.className = 'state';
    st.appendChild(stateMark(r.kind));
    const lab = document.createElement('span');
    lab.textContent = isFR() ? FR().ui.states[r.kind] : r.kind;
    st.appendChild(lab);
    const row = document.createElement('div');
    row.className = 'ledger-row';
    row.append(dt, dd, st);
    root.appendChild(row);
  });
}

/* --------------------------------------------------------------------- bindings --- */
function bind(data) {
  const best = data.coverage
    .filter((c) => c.exchangeable && c.coverage != null && c.coverage < 0.99)
    .sort((a, b) => Math.abs(a.coverage - 0.9) - Math.abs(b.coverage - 0.9))[0];
  const shifted = data.coverage.find((c) => !c.exchangeable);
  const acc = (m, s) => {
    const row = data.accuracy.find((x) => x.model === m);
    return row && row.by_split[s] ? row.by_split[s].value : null;
  };
  const fields = {
    nominalPct: (data.verdicts.h2.nominal * 100).toFixed(0),
    bestCoverage: best ? pct(best.coverage) : '—',
    shiftedCoverage: shifted ? pct(shifted.coverage) : '—',
    shiftedAuroc: shifted ? fmt(shifted.auroc) : '—',
    rankChanges: data.verdicts.h1b.changes,
    rankOf: data.verdicts.h1b.of,
    flowContext: fmt(acc('pertflow', 'unseen_context')),
    ridgeContext: fmt(acc('ridge_embedding', 'unseen_context')),
    mlpContext: fmt(acc('pertflow_mlp', 'unseen_context')),
    blindPct: pct(1 - data.verdicts.h3.blind_regret),
  };
  for (const [k, val] of Object.entries(fields)) {
    document.querySelectorAll(`[data-field="${k}"]`).forEach((n) => { n.textContent = val; });
  }
  const prov = document.getElementById('provenance-line');
  if (prov) {
    prov.textContent = isFR()
      ? FR().ui.provenance(data.meta.provenance, data.meta.seeds, data.meta.generated_from, data.meta.git_commit)
      : `${data.meta.provenance} data · ${data.meta.seeds} seeds · generated from ${data.meta.generated_from}`
        + (data.meta.git_commit ? ` · commit ${data.meta.git_commit}` : '');
  }
}

/* ------------------------------------------------------------------------ boot --- */
function reveal() {
  if (!('IntersectionObserver' in window)) {
    document.querySelectorAll('.reveal').forEach((n) => n.classList.add('in'));
    return;
  }
  const io = new IntersectionObserver((entries) => {
    entries.forEach((e) => { if (e.isIntersecting) { e.target.classList.add('in'); io.unobserve(e.target); } });
  }, { rootMargin: '0px 0px -8% 0px' });
  document.querySelectorAll('.reveal').forEach((n) => io.observe(n));
}

/* Everything the charts, tables and ledger draw is regenerated on a language switch,
   because each builds its own text rather than reading it from the document. */
function renderGenerated(data) {
  ['chart-coverage', 'chart-accuracy', 'table-accuracy', 'table-search', 'ledger']
    .forEach((id) => { const n = document.getElementById(id); if (n) n.innerHTML = ''; });
  bind(data);
  coverageChart(document.getElementById('chart-coverage'), data);
  accuracyChart(document.getElementById('chart-accuracy'), data);
  accuracyTable(document.getElementById('table-accuracy'), data);
  searchTable(document.getElementById('table-search'), data);
  ledger(document.getElementById('ledger'), data);
}

/* The English copy stays in the document and is stashed on first switch, so returning
   to English restores the original markup rather than a re-translation of it. */
function applyLang(data) {
  const fr = isFR();
  document.documentElement.lang = fr ? 'fr' : 'en';
  document.querySelectorAll('[data-i18n]').forEach((n) => {
    if (n.dataset.en === undefined) n.dataset.en = n.innerHTML;
    const val = fr ? (FR().html[n.dataset.i18n] ?? n.dataset.en) : n.dataset.en;
    n.innerHTML = val;
  });
  if (document.documentElement.dataset.titleEn === undefined) {
    document.documentElement.dataset.titleEn = document.title;
  }
  document.title = fr ? (FR().title || document.title) : document.documentElement.dataset.titleEn;
  const btn = document.getElementById('lang-toggle');
  if (btn) {
    btn.textContent = fr ? 'EN' : 'FR';
    btn.setAttribute('aria-label', fr ? 'Switch to English' : 'Passer en français');
  }
  document.querySelectorAll('#knockdowns .knock').forEach((b) => {
    const r = REGULATORS.find((x) => x.id === b.dataset.reg);
    if (!r) return;
    b.textContent = fr ? FR().ui.kdLabel(r.id) : r.label;
    b.title = fr ? (r.drives === 'res' ? FR().ui.kdResistance : FR().ui.kdSensitivity)
                 : (r.drives === 'res' ? 'Resistance driver' : 'Sensitivity driver');
  });
  if (data) renderGenerated(data);
}

async function boot() {
  sizeHero();
  redrawInteractive();
  let data;
  try {
    data = await (await fetch('data.json')).json();
  } catch (err) {
    console.error('could not load data.json', err);
    return;
  }
  applyLang(data);
  const btn = document.getElementById('lang-toggle');
  if (btn) {
    btn.addEventListener('click', () => {
      LANG = isFR() ? 'en' : 'fr';
      try { localStorage.setItem('lang', LANG); } catch { /* private mode */ }
      applyLang(data);
    });
  }
  document.querySelectorAll('section .wrap > *').forEach((n, i) => {
    if (i % 1 === 0) n.classList.add('reveal');
  });
  reveal();
}

let resizeTimer;
window.addEventListener('resize', () => {
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(() => { sizeHero(); redrawInteractive(); }, 160);
});

boot();
