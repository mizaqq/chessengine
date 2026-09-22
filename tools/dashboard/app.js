/* chessengine live dashboard: polls /api/state, keeps UI state across polls. */
const $ = id => document.getElementById(id);
const fmt = (v, nd = 3) => v == null ? "–" : (typeof v === "number" ? v.toFixed(nd) : String(v));
const age = s => s >= 3600 ? `${Math.floor(s / 3600)}h ${String(Math.floor(s % 3600 / 60)).padStart(2, "0")}m` : `${Math.floor(s / 60)}m ${String(Math.floor(s % 60)).padStart(2, "0")}s`;
const when = ts => { const d = new Date(ts * 1000); return `${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")} ${String(d.getHours()).padStart(2, "0")}:${String(d.getMinutes()).padStart(2, "0")}`; };
const PALETTE = ["#2f5d8a", "#c7522e", "#2e7d32", "#7b3fa0", "#b7791f", "#0f8b8d", "#8a2f5d", "#555"];
const color = i => PALETTE[i % PALETTE.length];

const S = { seen: {}, state: null, sort: { key: "mtime", dir: -1 }, selected: new Set(), runCache: {}, charts: {}, matrixFile: null, outFile: null,
            games: {}, gi: null, mi: 0, timer: null, metrics: new Set(["lichess_top1"]) };
const DASHES = [[], [6, 3], [2, 2], [8, 3, 2, 3], [1, 3], [10, 4], [4, 2, 1, 2], [12, 3, 3, 3]];

const COLS = [
  ["name", "arm", "l"], ["mtime", "when"], ["updates", "upd"], ["elapsed_min", "min"], ["filters", "net"],
  ["m1", "m1"], ["m2", "m2"], ["m3", "m3"], ["m4", "m4"], ["end", "end"], ["prior", "prior"], ["entropy", "entropy"],
  ["own_m1", "own m1"], ["decisive", "decisive"], ["elo", "elo"], ["fin0", "fin d2"], ["fin1", "d10"], ["fin2", "d20"], ["init_from", "from", "l"]];
const val = (r, k) => k in r ? r[k] : (k in r.dials ? r.dials[k] : k.startsWith("fin") ? r.finish[+k[3]] : null);


/* ---------------- legend: what each metric means (plain words) */
const INFO = {
  name: "experiment folder / arm", when: "when run.json was written", updates: "training updates in this run (each = one rollout of 16 boards x 64 plies + PPO)",
  elapsed_min: "wall-clock minutes of training", filters: "network width (filters); ·gpu = trained on the Apple GPU", init_from: "checkpoint this run continued from",
  m1: "held-out Lichess mate-in-1 puzzles: share where the network's top move mates (2,000 positions)",
  m2: "held-out mate-in-2: top move is the verified forcing first move", m3: "held-out mate-in-3, same rule", m4: "held-out mate-in-4, same rule",
  end: "held-out Lichess endgame puzzles that end in mate: top move matches the solution", prior: "score vs the supervised prior in 20 sampled games at the end (0.5 = par)",
  entropy: "policy entropy on game boards, normalised by log(legal moves); 1 = uniform, 0 = deterministic", own_m1: "own games (40): share of mate-in-1 chances the network actually took",
  decisive: "own games: share that ended in a win rather than a draw / move cap",
  elo: "Stockfish UCI_Elo ladder fit (CCRL blitz scale, floor 1320): logistic MLE with half the 95% bootstrap width; * = bound / extrapolation, see elo.json", fin0: "held-out finishing: rewound 2 plies before a human mate, network converts (100 games)",
  fin1: "same at 10 plies before the mate", fin2: "same at 20 plies before the mate",
  lichess_top1: "= m1 (held-out mate-in-1 top-1)", lichess_m2_top1: "= m2", lichess_m3_top1: "= m3", lichess_m4_top1: "= m4", lichess_end_top1: "= end (endgame mates)",
  prior_score: "score vs the supervised prior, 20 games per eval", puzzle_solved_rate: "training puzzle boards solved this update (all depths)", puzzle_solved_rate_m1: "training mate-in-1 boards solved",
  finish_success_rate: "training finishing boards converted (human starts, all depths)", finish_depth: "current depth of the finishing curriculum (max plies before the mate a board may start)",
  technique_success_rate_Q: "training queen-vs-king boards won, including wins made by the rules demonstrator", technique_clean_success_rate_Q: "same, only episodes where the demonstrator never acted (drives the curriculum)",
  technique_rate_Q: "held-out queen-vs-king at the hardest level: network alone, 50 games", mean_entropy_normalized_game: "= entropy (game boards)", mean_entropy_normalized_puzzle: "entropy on puzzle boards",
  mean_value_loss: "value-head MSE against the return targets (PPO)", mean_policy_loss: "PPO clipped surrogate (negative = pushing good moves up)", mean_approx_kl: "mean(old_logp - new_logp): how far one update moved the policy; ~0.03-0.05 healthy",
  mean_clip_fraction: "share of samples whose ratio left the [0.8, 1.2] band; ~0.2 healthy, 0.5 = trust band is a fiction", draw_rate: "training games drawn (incl. the 120-ply cap)",
  mean_terminal_return: "mean terminal reward of finished training games (win +2, loss -2, draw -0.5)", sil_positive_share: "self-imitation buffer: share of plies whose past return still beats the value head (what is left to learn)",
  pool_score: "learner's score vs the frozen pool opponents on the 5 pool boards", punish_moves: "learner plies where a punishing move (mate / free piece) existed on punish boards", guide_demo_share: "guided moves that were the demonstrator's pick"
};
function renderLegend() {
  const cols = COLS.map(([k, lbl]) => `<tr><td class="l"><b>${lbl}</b></td><td class="l">${INFO[k] || ""}</td></tr>`).join("");
  const mets = METRICS.filter(m => !COLS.some(([k]) => INFO[m] && INFO[m].startsWith("= "))).map(m => `<tr><td class="l"><b>${m}</b></td><td class="l">${INFO[m] || ""}</td></tr>`).join("");
  $("legend").innerHTML = `<table><tr><th class="l">results column</th><th class="l">meaning</th></tr>${cols}</table><br><table><tr><th class="l">curve metric</th><th class="l">meaning</th></tr>${mets}</table>`;
}

/* ---------------- charts */
function lineChart(id, opts = {}) {
  if (S.charts[id]) return S.charts[id];
  const c = new Chart($(id), { type: "line", data: { datasets: [] },
    options: { animation: false, responsive: true, maintainAspectRatio: false, parsing: false, normalized: true,
      plugins: { legend: { display: opts.legend !== false, labels: { boxWidth: 10, font: { size: 11 } } }, tooltip: { mode: "nearest", intersect: false } },
      elements: { point: { radius: opts.points ? 2 : 0 } },
      scales: { x: { type: "linear", title: { display: !!opts.xTitle, text: opts.xTitle }, ticks: { font: { size: 10 } } },
                y: { ticks: { font: { size: 10 } }, min: opts.ymin, max: opts.ymax } } } });
  S.charts[id] = c; return c;
}
function setData(chart, datasets) { chart.data.datasets = datasets; chart.update("none"); }
const series = (logs, key) => logs.filter(e => e[key] != null).map(e => ({ x: e.episode, y: e[key] }));

/* ---------------- ETA: from the run's own start time when the progress file carries it,
   otherwise from the rate observed since this page first saw the run */
function etaFor(arm, done, total, started, updated, now) {
  const seen = S.seen[arm] || (S.seen[arm] = { t: updated, n: done });
  let rate = null;                                   // units per second
  if (started && done > 0) rate = done / Math.max(updated - started, 1);
  else if (done > seen.n && updated > seen.t) rate = (done - seen.n) / (updated - seen.t);
  if (!rate) return { text: "", rate: "" };
  const perMin = rate * 60;
  const out = { rate: `${perMin >= 10 ? perMin.toFixed(0) : perMin.toFixed(1)} / min` };
  if (total && done < total) {
    const left = (total - done) / rate;
    const end = new Date((now + left) * 1000);
    out.text = `≈ ${age(left)} left · ends ${String(end.getHours()).padStart(2, "0")}:${String(end.getMinutes()).padStart(2, "0")}`;
  }
  if (started) out.rate += ` · running ${age(updated - started)}`;
  return out;
}

/* ---------------- running */
function renderRunning(st) {
  const box = $("running"); const cards = [];
  const stale = upd => st.now - upd > 900;
  for (const r of st.running) cards.push(`<div class="card run"><b>${r.task}</b><span class="tag">${r.driver}</span> <span class="sub small">started ${r.started}</span><div class="sub small">${r.last_line || ""}</div></div>`);
  for (const p of st.progress) {
    const pct = (p.episode || 0) / Math.max(p.episodes || 1, 1) * 100;
    const eta = etaFor(p.arm, p.episode, p.episodes, p.started, p.updated, st.now);
    const dials = Object.entries(p.dials).filter(([k, v]) => v != null).map(([k, v]) => `<span>${k} <b>${fmt(v)}</b></span>`).join("");
    const id = "prog_" + p.arm.replace(/[^\w]/g, "_");
    cards.push(`<div class="card run ${stale(p.updated) ? "stale" : ""}"><b>${p.arm}</b><span class="tag">update ${p.episode} / ${p.episodes}</span>${eta.text ? `<span class="tag">${eta.text}</span>` : ""}
      <span class="sub small">last write ${age(st.now - p.updated)} ago${stale(p.updated) ? " · stale?" : ""}${eta.rate ? ` · ${eta.rate}` : ""}</span>
      <div class="bar"><div style="width:${pct.toFixed(0)}%"></div></div><div class="dials">${dials}</div>
      <div class="mini">${["lichess_top1", "mean_entropy_normalized_game", "finish_success_rate"].map(k => `<div><div class="sub small">${k}</div><canvas id="${id}_${k}"></canvas></div>`).join("")}</div></div>`);
  }
  for (const p of st.running_pretraining) {
    const id = "pre_" + p.arm.replace(/[^\w]/g, "_");
    const eta = etaFor(p.arm, p.steps, null, null, p.updated, st.now);
    cards.push(`<div class="card run ${stale(p.updated) ? "stale" : ""}"><b>${p.arm}</b><span class="tag">step ${p.steps}</span> <span class="sub small">last write ${age(st.now - p.updated)} ago${eta.rate ? ` · ${eta.rate}` : ""}</span>
      <div class="dials"><span>held-out top-1 <b>${fmt(p.last.eval_top1)}</b></span><span>train top-1 <b>${fmt(p.last.train_top1)}</b></span><span>eval CE <b>${fmt(p.last.eval_ce)}</b></span></div>
      <div class="mini" style="grid-template-columns:1fr"><canvas id="${id}" style="height:90px!important"></canvas></div></div>`);
  }
  if (!cards.length) cards.push(`<div class="sub">nothing running</div>`);
  box.innerHTML = cards.join("");
  // charts must be created after the canvases exist; chart objects are per canvas so drop old ones
  for (const k of Object.keys(S.charts)) if (k.startsWith("prog_") || k.startsWith("pre_")) { S.charts[k].destroy(); delete S.charts[k]; }
  for (const p of st.progress) {
    const id = "prog_" + p.arm.replace(/[^\w]/g, "_");
    for (const k of ["lichess_top1", "mean_entropy_normalized_game", "finish_success_rate"]) {
      const pts = (p.curves[k] || []).map(([x, y]) => ({ x, y })); if (pts.length < 2) continue;
      setData(lineChart(`${id}_${k}`, { legend: false }), [{ data: pts, borderColor: color(0), borderWidth: 1.5 }]);
    }
  }
  for (const p of st.running_pretraining) {
    const id = "pre_" + p.arm.replace(/[^\w]/g, "_");
    const ref = st.pretraining.find(q => q.arm === "human-pretraining/sl");
    const ds = [{ label: p.arm + " held-out", data: p.curve.map(([x, y]) => ({ x, y })), borderColor: color(0), borderWidth: 1.5 }];
    if (ref) ds.push({ label: "sl (128, small data)", data: ref.curve.map(([x, y]) => ({ x, y })), borderColor: "#999", borderDash: [4, 3], borderWidth: 1 });
    setData(lineChart(id, { legend: true }), ds);
  }
}

/* ---------------- results */
function renderResults(st) {
  const f = $("filter").value.toLowerCase(); const onlyM = $("onlyModels").checked;
  let rows = st.results.filter(r => (!f || r.name.toLowerCase().includes(f)) && (!onlyM || r.has_model));
  const { key, dir } = S.sort;
  rows.sort((a, b) => { const x = val(a, key), y = val(b, key); if (x == null && y == null) return 0; if (x == null) return 1; if (y == null) return -1;
    return (typeof x === "number" ? x - y : String(x).localeCompare(String(y))) * dir; });
  const head = `<tr><th></th>${COLS.map(([k, lbl, cls]) => `<th class="${cls || ""} ${key === k ? "sorted" : ""}" data-k="${k}" title="${(INFO[k] || "").replace(/"/g, "'")}">${lbl}${key === k ? (dir > 0 ? " ▲" : " ▼") : ""}</th>`).join("")}</tr>`;
  const body = rows.map(r => `<tr><td><input type="checkbox" data-arm="${r.name}" ${S.selected.has(r.name) ? "checked" : ""}></td>${COLS.map(([k, , cls]) => {
    let v = val(r, k);
    if (k === "mtime") v = when(v); else if (k === "elapsed_min") v = v.toFixed(0); else if (k === "init_from") v = v ? v.split("/").pop() : "";
    else if (k === "filters") v = `${r.filters}${r.device && r.device !== "cpu" ? "·gpu" : ""}`;
    else if (k === "elo") v = v == null ? "–" : `${v}${r.elo_bound ? "*" : ""}<span class="sub"> ±${r.elo_ci ? Math.round((r.elo_ci[1] - r.elo_ci[0]) / 2) : "?"}</span>`; else if (k === "decisive" || k.startsWith("fin")) v = fmt(v, 2); else if (typeof v === "number") v = fmt(v);
    const sel = S.selected.has(r.name) ? `style="border-left:3px solid ${color([...S.selected].indexOf(r.name))}"` : "";
    return `<td class="${cls || ""}" ${k === "name" ? sel : ""}>${v ?? "–"}</td>`; }).join("")}</tr>`).join("");
  $("results").innerHTML = head + body;
  $("results").querySelectorAll("th[data-k]").forEach(th => th.onclick = () => { const k = th.dataset.k; S.sort = { key: k, dir: S.sort.key === k ? -S.sort.dir : (k === "name" || k === "init_from" ? 1 : -1) }; renderResults(S.state); });
  $("results").querySelectorAll("input[data-arm]").forEach(cb => cb.onchange = () => { cb.checked ? S.selected.add(cb.dataset.arm) : S.selected.delete(cb.dataset.arm); renderResults(S.state); renderCurves(); });
}

/* ---------------- curves */
const METRICS = ["lichess_top1", "lichess_m2_top1", "lichess_m3_top1", "lichess_m4_top1", "lichess_end_top1", "prior_score", "puzzle_solved_rate", "puzzle_solved_rate_m1",
  "finish_success_rate", "finish_depth", "technique_success_rate_Q", "technique_clean_success_rate_Q", "technique_rate_Q", "mean_entropy_normalized_game", "mean_entropy_normalized_puzzle",
  "mean_value_loss", "mean_policy_loss", "mean_approx_kl", "mean_clip_fraction", "draw_rate", "mean_terminal_return", "sil_positive_share", "pool_score", "punish_moves", "guide_demo_share"];
async function getRun(arm) { if (!S.runCache[arm]) S.runCache[arm] = await (await fetch(`/api/run?arm=${encodeURIComponent(arm)}`)).json(); return S.runCache[arm]; }
async function renderCurves() {
  const arms = [...S.selected], mets = [...S.metrics]; const chart = lineChart("curves", { legend: true, xTitle: "update" });
  if (!arms.length || !mets.length) { setData(chart, []); $("curvesHint").textContent = arms.length ? "pick at least one metric" : "select arms in the results table"; return; }
  const align = $("alignX").checked, norm = $("normY").checked; const ds = [];
  for (const [i, arm] of arms.entries()) {
    const run = await getRun(arm);
    for (const [j, met] of mets.entries()) {
      let pts = series(run.logs, met); if (!pts.length) continue;
      if (align) { const x0 = pts[0].x; pts = pts.map(p => ({ x: p.x - x0, y: p.y })); }
      if (norm) { const ys = pts.map(p => p.y), lo = Math.min(...ys), hi = Math.max(...ys); pts = pts.map(p => ({ x: p.x, y: hi > lo ? (p.y - lo) / (hi - lo) : 0.5 })); }
      ds.push({ label: `${arm} · ${met}`, data: pts, borderColor: color(i), borderDash: DASHES[j % DASHES.length], borderWidth: 1.6, pointRadius: pts.length < 30 ? 2 : 0 });
    }
  }
  setData(chart, ds);
  $("curvesHint").textContent = `${arms.length} arm(s) × ${mets.length} metric(s)`;
}
function renderMetricInfo() {
  $("metricInfo").innerHTML = [...S.metrics].map((m, j) => `<div><span class="pill" style="background:#555;border-radius:0;height:2px;width:22px;${DASHES[j % DASHES.length].length ? "background:repeating-linear-gradient(90deg,#555 0 4px,transparent 4px 7px)" : ""}"></span><b>${m}</b> — ${INFO[m] || ""}</div>`).join("");
}
function renderMetricChips() {
  renderMetricInfo();
  $("metrics").innerHTML = METRICS.map(m => `<label class="${S.metrics.has(m) ? "on" : ""}" title="${(INFO[m] || "").replace(/"/g, "'")}"><input type="checkbox" data-m="${m}" ${S.metrics.has(m) ? "checked" : ""}>${m}</label>`).join("");
  $("metrics").querySelectorAll("input[data-m]").forEach(cb => cb.onchange = () => { cb.checked ? S.metrics.add(cb.dataset.m) : S.metrics.delete(cb.dataset.m); renderMetricChips(); renderCurves(); });
}

/* ---------------- matrices */
function renderMatrix(st) {
  const sel = $("matrixSel"); const files = st.matrices.map(m => m.file);
  if (sel.options.length !== files.length || [...sel.options].some((o, i) => o.value !== files[i])) {
    sel.innerHTML = st.matrices.map(m => `<option value="${m.file}">${m.file} · ${when(m.mtime)}</option>`).join(""); }
  if (!S.matrixFile || !files.includes(S.matrixFile)) S.matrixFile = files[0]; sel.value = S.matrixFile;
  const m = st.matrices.find(x => x.file === S.matrixFile); if (!m) { $("matrix").innerHTML = ""; return; }
  const wl = (a, b) => { const k = JSON.stringify([a, b]), kr = JSON.stringify([b, a]); const w = m.wins_json[k]; if (w) return w; const r = m.wins_json[kr]; return r ? [r[1], r[0]] : null; };
  const heat = s => { const t = Math.max(-1, Math.min(1, (s - 0.5) * 4)); return t >= 0 ? `rgba(46,125,50,${(t * 0.45).toFixed(2)})` : `rgba(178,58,72,${(-t * 0.45).toFixed(2)})`; };
  const means = Object.fromEntries(m.names.map(a => [a, m.names.filter(b => b !== a).reduce((s, b) => s + m.table[a][b], 0) / (m.names.length - 1)]));
  $("matrix").innerHTML = `<table class="heat"><tr><th></th>${m.names.map(n => `<th>${n}</th>`).join("")}<th>mean</th></tr>` + m.names.map(a => `<tr><td class="l"><b>${a}</b></td>` +
    m.names.map(b => a === b ? `<td class="sub">–</td>` : `<td style="background:${heat(m.table[a][b])}">${m.table[a][b].toFixed(2)}${wl(a, b) ? `<span>${wl(a, b)[0]}–${wl(a, b)[1]}</span>` : ""}</td>`).join("") + `<td><b>${means[a].toFixed(3)}</b></td></tr>`).join("") + "</table>";
}

/* ---------------- pretraining */
function renderPre(st) {
  const chart = lineChart("pre", { legend: true, xTitle: "batch", ymin: 0 });
  setData(chart, st.pretraining.map((p, i) => ({ label: p.arm.split("/").pop() + (p.done ? "" : " (running)"), data: p.curve.map(([x, y]) => ({ x, y })), borderColor: color(i), borderWidth: 1.5, borderDash: p.done ? [] : [5, 3] })));
  $("preTable").innerHTML = `<table><tr><th class="l">run</th><th>steps</th><th>held-out top-1</th><th>eval CE</th><th>status</th></tr>` +
    st.pretraining.map(p => `<tr><td class="l">${p.arm.split("/").pop()}</td><td>${p.steps ?? "–"}</td><td>${fmt(p.final ? p.final.eval_top1 : p.last.eval_top1)}</td><td>${fmt(p.final ? p.final.eval_ce : p.last.eval_ce)}</td><td>${p.done ? "done" : "running"}</td></tr>`).join("") + "</table>";
}

/* ---------------- outcomes */
async function renderOutcomes(st) {
  const sel = $("outSel"); const files = st.outcomes.map(o => o.file);
  if ([...sel.options].map(o => o.value).join() !== files.join()) sel.innerHTML = st.outcomes.map(o => `<option value="${o.file}">${o.file} · ${when(o.mtime)}</option>`).join("");
  if (!S.outFile || !files.includes(S.outFile)) { S.outFile = files[0]; await loadOutcome(); } sel.value = S.outFile;
}
async function loadOutcome() { if (!S.outFile) return; const d = await (await fetch(`/api/outcome?file=${encodeURIComponent(S.outFile)}`)).json(); $("outText").textContent = d.text || d.error; }

/* ---------------- games */
const U = { K: "♔", Q: "♕", R: "♖", B: "♗", N: "♘", P: "♙", k: "♚", q: "♛", r: "♜", b: "♝", n: "♞", p: "♟" };
const START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
const fenBoard = fen => fen.split(" ")[0].split("/").map(r => { const row = []; for (const c of r) { if (/\d/.test(c)) for (let i = 0; i < +c; i++) row.push(""); else row.push(c); } return row; });
async function refreshGames(st) {
  const ids = st.games.map(g => g.id);
  if (ids.some(id => !S.games[id])) S.games = await (await fetch("/api/games")).json();
  const sel = $("gsel");
  if ([...sel.options].map(o => o.value).join() !== ids.join()) {
    sel.innerHTML = st.games.map(g => `<option value="${g.id}">${g.arm} · game ${g.n} · ${g.result} · ${g.plies} plies</option>`).join("");
    if (!S.gi || !ids.includes(S.gi)) { S.gi = ids[0]; S.mi = 0; } sel.value = S.gi; drawBoard();
  }
}
function drawBoard() {
  const g = S.games[S.gi]; if (!g) { $("board").textContent = "no games yet"; return; }
  const fen = S.mi === 0 ? START : g.moves[S.mi - 1].fen; const b = fenBoard(fen);
  const prev = fenBoard(S.mi <= 1 ? START : g.moves[S.mi - 2].fen);
  const el = $("board"); el.innerHTML = "";
  for (let r = 0; r < 8; r++) for (let c = 0; c < 8; c++) { const d = document.createElement("div"); d.className = "sq " + (((r + c) % 2) ? "dk" : "lt");
    if (S.mi > 0) { if (prev[r][c] && !b[r][c]) d.classList.add("from"); if (b[r][c] && prev[r][c] !== b[r][c]) d.classList.add("to"); }
    d.textContent = U[b[r][c]] || ""; el.appendChild(d); }
  $("slider").max = g.moves.length; $("slider").value = S.mi;
  const m = S.mi > 0 ? g.moves[S.mi - 1] : null;
  $("mv").textContent = S.mi === 0 ? "start position" : `ply ${S.mi}/${g.moves.length}: ${m.side} ${m.san}`;
  let rows = `<tr><th>arm</th><td>${g.arm}</td></tr><tr><th>result</th><td class="${g.result.includes("win") ? "win" : ""}">${g.result}</td></tr>`;
  if (m) { rows += `<tr><th>played</th><td>${m.san} <span class="sub">(p ${m.p})</span></td></tr><tr><th>value</th><td>${m.value}</td></tr>`;
    m.top.forEach((t, i) => rows += `<tr><th>${i ? "" : "top-3"}</th><td>${t[0]} <span class="sub">${t[1]}</span></td></tr>`); }
  $("info").innerHTML = rows;
  const chart = lineChart("valplot", { legend: false, xTitle: "", ymin: -2, ymax: 2 });
  setData(chart, [{ data: g.moves.map((x, i) => ({ x: i + 1, y: x.value })), borderColor: color(0), borderWidth: 1.4 },
                  { data: [{ x: Math.max(S.mi, 1), y: -2 }, { x: Math.max(S.mi, 1), y: 2 }], borderColor: "#c7522e", borderWidth: 1 }]);
}
function go(n) { const g = S.games[S.gi]; if (!g) return; S.mi = Math.max(0, Math.min(g.moves.length, n)); drawBoard(); }
$("gsel").onchange = () => { S.gi = $("gsel").value; S.mi = 0; drawBoard(); };
$("first").onclick = () => go(0); $("prev").onclick = () => go(S.mi - 1); $("next").onclick = () => go(S.mi + 1); $("last").onclick = () => go(1e9);
$("slider").oninput = e => go(+e.target.value);
$("play").onclick = function () { if (S.timer) { clearInterval(S.timer); S.timer = null; this.textContent = "▶ play"; return; }
  this.textContent = "⏸ pause"; S.timer = setInterval(() => { const g = S.games[S.gi]; if (!g || S.mi >= g.moves.length) { clearInterval(S.timer); S.timer = null; $("play").textContent = "▶ play"; return; } go(S.mi + 1); }, 600); };
document.addEventListener("keydown", e => { if (e.target.tagName === "INPUT") return; if (e.key === "ArrowRight") go(S.mi + 1); if (e.key === "ArrowLeft") go(S.mi - 1); });
$("playMore").onclick = async () => { const g = S.games[S.gi]; if (!g) return; const r = await (await fetch(`/api/play?arm=${encodeURIComponent(g.arm)}`, { method: "POST" })).json(); $("playMsg").textContent = r.queued ? `queued for ${r.queued}; appears when played` : (r.error || ""); };

/* ---------------- wiring */
renderMetricChips(); renderLegend();
$("normY").onchange = renderCurves;
$("alignX").onchange = renderCurves;
$("filter").oninput = () => renderResults(S.state); $("onlyModels").onchange = () => renderResults(S.state);
$("clearSel").onclick = () => { S.selected.clear(); renderResults(S.state); renderCurves(); };
$("matrixSel").onchange = () => { S.matrixFile = $("matrixSel").value; renderMatrix(S.state); };
$("outSel").onchange = () => { S.outFile = $("outSel").value; loadOutcome(); };

async function poll() {
  try {
    const st = await (await fetch("/api/state")).json();
    for (const m of st.matrices) m.wins_json = Object.fromEntries(Object.entries(m.wins).map(([k, v]) => [k, v]));
    S.state = st;
    const nRun = st.running.length + st.progress.length + st.running_pretraining.length;
    $("status").textContent = `updated ${new Date().toLocaleTimeString()} · ${nRun} running · ${st.results.length} arms`;
    renderRunning(st); renderResults(st); renderMatrix(st); renderPre(st); await renderOutcomes(st); await refreshGames(st);
    // arms in progress get fresh logs each poll
    for (const p of st.progress) delete S.runCache[p.arm];
    if ([...S.selected].some(a => st.progress.find(p => p.arm === a))) renderCurves();
  } catch (e) { $("status").innerHTML = `<span class="err">connection lost: ${e.message}</span>`; }
  setTimeout(poll, 5000);
}
poll();
