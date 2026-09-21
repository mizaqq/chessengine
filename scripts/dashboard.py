"""Local dashboard: running tasks with live curves, results of every arm, win matrices,
outcome notes, and a replay viewer for games played by the recent checkpoints.

    python -m scripts.dashboard [--out dashboard/index.html] [--checkpoints 3] [--games 2]
                                [--watch 60] [--open]

Reads experiments/**/run.json, games.json, finishes.json, progress.json (written every log
interval by training), experiments/**/run_*.log (driver logs: "=== NAME start" / "done"),
experiments/win-matrix/matrix*.json and experiments/*/outcome.md. Games are played on demand
with src.viz.play.play_game and cached in dashboard/games_cache.json by checkpoint path.
Static HTML, no external resources; with --watch the page refreshes itself.
"""
import argparse
import glob
import html
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments"

DIALS = [("lichess_top1", "m1"), ("lichess_m2_top1", "m2"), ("lichess_m3_top1", "m3"), ("lichess_m4_top1", "m4"),
         ("lichess_end_top1", "end"), ("prior_score", "prior"), ("mean_entropy_normalized_game", "entropy")]
CURVES = ["lichess_top1", "puzzle_solved_rate", "finish_success_rate", "mean_entropy_normalized_game",
          "mean_value_loss", "mean_approx_kl"]


def _load(p):
    try:
        return json.loads(Path(p).read_text())
    except Exception:
        return None


def _fmt(v, nd=3):
    if v is None:
        return "–"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def _age(ts):
    s = max(0, time.time() - ts)
    return f"{int(s // 3600)}h {int(s % 3600 // 60):02d}m" if s >= 3600 else f"{int(s // 60)}m {int(s % 60):02d}s"


# ---------------------------------------------------------------- data collection

def collect_results():
    rows = []
    for rj in sorted(EXP.glob("*/*/run.json")):
        d = _load(rj)
        if not d:
            continue
        arm = rj.parent
        last = d["logs"][-1] if d.get("logs") else {}
        cfg = d.get("config", {})
        g = _load(arm / "games.json") or {}
        f = _load(arm / "finishes.json") or {}
        rows.append({
            "name": f"{arm.parent.name}/{arm.name}", "mtime": rj.stat().st_mtime,
            "updates": cfg.get("max_updates"), "init_from": cfg.get("init_from") or "",
            "elapsed_min": (d.get("elapsed_s") or 0) / 60,
            "dials": {short: last.get(k) for k, short in DIALS},
            "own_m1": g.get("mate_in_one_rate"), "decisive": g.get("decisive_rate"),
            "finish": [f.get(f"finish_rate_d{k}") for k in (2, 10, 20)],
            "device": cfg.get("device", "cpu"), "filters": cfg.get("num_filters", 128),
            "has_model": bool(list(arm.glob("model_*.pth"))),
        })
    rows.sort(key=lambda r: -r["mtime"])
    return rows


def collect_running():
    """Driver logs with a start line whose matching 'done' has not appeared, plus progress
    files touched in the last 2 hours whose arm has no run.json yet."""
    running = []
    for log in EXP.glob("*/run_*.log"):
        try:
            text = log.read_text()
        except Exception:
            continue
        starts = re.findall(r"^=== (\S+) start (\d\d:\d\d:\d\d)", text, re.M)
        dones = set(re.findall(r"^=== (\S+) (?:all )?done", text, re.M))
        for name, hhmm in starts:
            if name not in dones and time.time() - log.stat().st_mtime < 24 * 3600:
                running.append({"task": name, "driver": log.name, "started": hhmm, "log": str(log.relative_to(ROOT)),
                                "last_line": text.strip().splitlines()[-1][:160] if text.strip() else ""})
    progress = []
    for pj in EXP.glob("*/*/progress.json"):
        if (pj.parent / "run.json").exists() and (pj.parent / "run.json").stat().st_mtime >= pj.stat().st_mtime:
            continue
        if time.time() - pj.stat().st_mtime > 2 * 3600:
            continue
        d = _load(pj)
        if not d:
            continue
        logs = d.get("logs", [])
        last = logs[-1] if logs else {}
        progress.append({
            "arm": f"{pj.parent.parent.name}/{pj.parent.name}", "episode": d.get("episode"), "episodes": d.get("episodes"),
            "updated": d.get("updated", pj.stat().st_mtime), "dials": {short: last.get(k) for k, short in DIALS},
            "curves": {k: [(e["episode"], e[k]) for e in logs if e.get(k) is not None] for k in CURVES},
        })
    return running, progress


def collect_matrices(limit=4):
    out = []
    for mj in sorted(EXP.glob("win-matrix/matrix*.json"), key=lambda p: -p.stat().st_mtime)[:limit]:
        d = _load(mj)
        if d:
            wins = {}
            for c in d.get("counts", []):
                cc = c["counts"]
                w = cc.get("a_white_white_win", 0) + cc.get("a_black_black_win", 0)
                l = cc.get("a_white_black_win", 0) + cc.get("a_black_white_win", 0)
                wins[(c["a"], c["b"])] = (w, l)
            out.append({"file": mj.name, "mtime": mj.stat().st_mtime, "names": d["names"], "table": d["table"], "wins": wins})
    return out


def collect_outcomes(limit=4, tail=30):
    out = []
    for om in sorted(EXP.glob("*/outcome.md"), key=lambda p: -p.stat().st_mtime)[:limit]:
        lines = om.read_text().strip().splitlines()
        out.append({"file": str(om.relative_to(ROOT)), "mtime": om.stat().st_mtime, "tail": "\n".join(lines[-tail:])})
    return out


# ---------------------------------------------------------------- games

def play_games(checkpoint_dirs, games_per_ckpt, cache_path):
    cache = _load(cache_path) or {}
    changed = False
    for ck in checkpoint_dirs:
        models = sorted(glob.glob(str(ROOT / ck / "model_*.pth")))
        if not models:
            continue
        key = models[-1]
        if key in cache and len(cache[key]["games"]) >= games_per_ckpt:
            continue
        # Imports here so the dashboard works without torch for the read-only parts.
        import torch
        from src.model.checkpoints import load_models
        from src.viz.play import play_game
        w, b, oriented, _ = load_models(ROOT / ck)
        games = []
        for i in range(games_per_ckpt):
            torch.manual_seed(100 + i)
            g = play_game(w, b, oriented=oriented, seed=100 + i)
            games.append({"result": g.result, "moves": [
                {"san": m.san, "fen": m.fen_after, "side": m.side, "value": round(m.value, 3),
                 "top": [(s, round(p, 3)) for s, p in m.top_moves], "p": round(m.prob_played, 3)} for m in g.moves]})
        cache[key] = {"arm": ck, "games": games, "played": time.time()}
        changed = True
    if changed:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        Path(cache_path).write_text(json.dumps(cache))
    return cache


# ---------------------------------------------------------------- rendering

def spark(points, w=220, h=44):
    if len(points) < 2:
        return ""
    xs = [p[0] for p in points]; ys = [p[1] for p in points]
    x0, x1 = min(xs), max(xs); y0, y1 = min(ys), max(ys)
    if x1 == x0: x1 = x0 + 1
    if y1 == y0: y1 = y0 + 1e-9
    pts = " ".join(f"{(x - x0) / (x1 - x0) * (w - 4) + 2:.1f},{h - 2 - (y - y0) / (y1 - y0) * (h - 4):.1f}" for x, y in points)
    return (f'<svg class="spark" viewBox="0 0 {w} {h}" width="{w}" height="{h}"><polyline points="{pts}"/></svg>'
            f'<span class="sparklbl">{ys[0]:.3f} → <b>{ys[-1]:.3f}</b></span>')


def render(results, running, progress, matrices, outcomes, games_cache, watch):
    e = html.escape
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    parts = []
    parts.append(f"""<!doctype html><html><head><meta charset="utf-8"><title>chessengine dashboard</title>
{'<meta http-equiv="refresh" content="%d">' % watch if watch else ''}
<style>
:root{{--bg:#f6f4ee;--card:#fff;--ink:#1e1e1e;--mute:#6b6b6b;--line:#e2ded4;--acc:#2f5d8a;--good:#2e7d32;--bad:#b23a48}}
body{{margin:0;padding:20px 24px;background:var(--bg);color:var(--ink);font:14px/1.45 -apple-system,Helvetica,Arial,sans-serif}}
h1{{font-size:20px;margin:0 0 4px}} h2{{font-size:15px;margin:22px 0 8px;color:var(--acc);text-transform:uppercase;letter-spacing:.06em}}
.sub{{color:var(--mute)}} .card{{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px 14px;margin:8px 0}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:10px}}
table{{border-collapse:collapse;width:100%;font-size:13px}} th,td{{padding:4px 8px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}}
th:first-child,td:first-child{{text-align:left}} th{{color:var(--mute);font-weight:600;position:sticky;top:0;background:var(--card)}}
.wrap{{overflow-x:auto}} .tag{{display:inline-block;padding:1px 7px;border-radius:10px;font-size:11px;background:#eef3f8;color:var(--acc);margin-left:6px}}
.run{{border-left:4px solid var(--good)}} .spark polyline{{fill:none;stroke:var(--acc);stroke-width:1.6}} .sparklbl{{font-size:11px;color:var(--mute);margin-left:6px}}
.curve{{display:inline-block;margin:4px 14px 4px 0;vertical-align:top}} .curve div{{font-size:11px;color:var(--mute)}}
pre{{white-space:pre-wrap;font-size:12px;background:#faf9f6;padding:8px;border-radius:6px;max-height:320px;overflow:auto}}
details summary{{cursor:pointer;color:var(--acc)}}
/* board */
.viewer{{display:flex;gap:18px;flex-wrap:wrap;align-items:flex-start}}
.board{{display:grid;grid-template-columns:repeat(8,44px);grid-template-rows:repeat(8,44px);border:2px solid #5a4a3a;width:352px}}
.sq{{display:flex;align-items:center;justify-content:center;font-size:32px;line-height:1}} .l{{background:#efe0c4}} .d{{background:#b58863}}
.sq.from{{box-shadow:inset 0 0 0 3px #e0b04a}} .sq.to{{box-shadow:inset 0 0 0 3px #c7522e}}
.ctl button{{margin:2px;padding:4px 10px}} .info{{min-width:260px;max-width:420px}} .info td{{text-align:left}}
.val{{height:60px}} .win{{color:var(--good)}} .loss{{color:var(--bad)}}
</style></head><body>
<h1>chessengine <span class="sub">dashboard</span></h1>
<div class="sub">generated {now}{' · auto-refresh every %ds' % watch if watch else ''} · regenerate: <code>python -m scripts.dashboard</code></div>""")

    # --- running
    parts.append("<h2>Running</h2>")
    if not running and not progress:
        parts.append('<div class="card sub">nothing running</div>')
    for r in running:
        parts.append(f'<div class="card run"><b>{e(r["task"])}</b><span class="tag">{e(r["driver"])}</span> '
                     f'<span class="sub">started {e(r["started"])} · {e(r["log"])}</span><br><span class="sub">{e(r["last_line"])}</span></div>')
    for p in progress:
        pct = (p["episode"] or 0) / max(p["episodes"] or 1, 1) * 100
        dials = " · ".join(f"{k} {_fmt(v)}" for k, v in p["dials"].items() if v is not None)
        curves = "".join(f'<span class="curve"><div>{e(k)}</div>{spark(v)}</span>' for k, v in p["curves"].items() if len(v) > 1)
        parts.append(f'<div class="card run"><b>{e(p["arm"])}</b> <span class="tag">update {p["episode"]} / {p["episodes"]} · {pct:.0f}%</span> '
                     f'<span class="sub">last write {_age(p["updated"])} ago</span><div class="sub">{e(dials)}</div><div>{curves}</div></div>')

    # --- results
    parts.append("<h2>Results</h2><div class='card wrap'><table><tr><th>arm</th><th>when</th><th>upd</th><th>min</th><th>net</th>"
                 "<th>m1</th><th>m2</th><th>m3</th><th>m4</th><th>end</th><th>prior</th><th>entropy</th><th>own m1</th><th>decisive</th>"
                 "<th>fin d2</th><th>d10</th><th>d20</th><th>from</th></tr>")
    for r in results:
        d = r["dials"]
        when = datetime.fromtimestamp(r["mtime"]).strftime("%m-%d %H:%M")
        parts.append(f"<tr><td>{e(r['name'])}</td><td>{when}</td><td>{_fmt(r['updates'])}</td><td>{r['elapsed_min']:.0f}</td>"
                     f"<td>{r['filters']}{'·' + e(str(r['device'])) if r['device'] != 'cpu' else ''}</td>"
                     + "".join(f"<td>{_fmt(d[k])}</td>" for k in ("m1", "m2", "m3", "m4", "end", "prior", "entropy"))
                     + f"<td>{_fmt(r['own_m1'])}</td><td>{_fmt(r['decisive'], 2)}</td>"
                     + "".join(f"<td>{_fmt(v, 2)}</td>" for v in r["finish"])
                     + f"<td class='sub'>{e(Path(r['init_from']).name if r['init_from'] else '')}</td></tr>")
    parts.append("</table></div>")

    # --- matrices
    parts.append("<h2>Win matrices</h2><div class='grid'>")
    for m in matrices:
        names = m["names"]
        rows = "".join("<tr><td><b>%s</b></td>" % e(a) + "".join(
            ("<td class='sub'>–</td>" if a == b else
             f"<td>{m['table'][a][b]:.2f}<br><span class='sub'>{m['wins'].get((a, b), m['wins'].get((b, a), ('?', '?'))[::-1] if (b, a) in m['wins'] else ('?', '?'))[0]}–{m['wins'].get((a, b), m['wins'].get((b, a), ('?', '?'))[::-1] if (b, a) in m['wins'] else ('?', '?'))[1]}</span></td>")
            for b in names) + "</tr>" for a in names)
        parts.append(f"<div class='card wrap'><b>{e(m['file'])}</b> <span class='sub'>{datetime.fromtimestamp(m['mtime']).strftime('%m-%d %H:%M')} · row score vs column, wins–losses</span>"
                     f"<table><tr><th></th>{''.join('<th>%s</th>' % e(n) for n in names)}</tr>{rows}</table></div>")
    parts.append("</div>")

    # --- outcomes
    parts.append("<h2>Outcome notes</h2>")
    for o in outcomes:
        parts.append(f"<details class='card'><summary>{e(o['file'])} <span class='sub'>({datetime.fromtimestamp(o['mtime']).strftime('%m-%d %H:%M')}, last lines)</span></summary><pre>{e(o['tail'])}</pre></details>")

    # --- games
    games = []
    for key, entry in sorted(games_cache.items(), key=lambda kv: -kv[1]["played"]):
        for i, g in enumerate(entry["games"]):
            games.append({"label": f"{entry['arm']} · game {i + 1} · {g['result']} · {len(g['moves'])} plies", "result": g["result"], "moves": g["moves"]})
    parts.append("<h2>Game viewer</h2><div class='card'><div class='viewer'><div><select id='gsel'></select>"
                 "<div class='board' id='board'></div><div class='ctl'><button id='first'>⏮</button><button id='prev'>◀</button>"
                 "<button id='play'>▶ play</button><button id='next'>▶</button><button id='last'>⏭</button> "
                 "<input type='range' id='slider' min='0' value='0' style='width:200px'> <span id='mv' class='sub'></span></div>"
                 "<svg id='valplot' class='val' width='352' height='60'></svg><div class='sub'>value head (mover's view) along the game; marker = current ply</div></div>"
                 "<div class='info'><table id='info'></table><div class='sub' style='margin-top:8px'>Top-3 moves are the network's probabilities at the position <i>before</i> the move; "
                 "p is the probability of the move actually played (sampled). Value is the mover's value head at the same position (shaping offset not added back).</div></div></div></div>")
    parts.append("<script>const GAMES=" + json.dumps(games) + ";")
    parts.append(r"""
const U={K:'♔',Q:'♕',R:'♖',B:'♗',N:'♘',P:'♙',k:'♚',q:'♛',r:'♜',b:'♝',n:'♞',p:'♟'};
const START='rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1';
let gi=0, mi=0, timer=null;
const sel=document.getElementById('gsel');
GAMES.forEach((g,i)=>{const o=document.createElement('option');o.value=i;o.textContent=g.label;sel.appendChild(o);});
function fenBoard(fen){const rows=fen.split(' ')[0].split('/');const out=[];for(const r of rows){const row=[];for(const c of r){if(/\d/.test(c)){for(let i=0;i<+c;i++)row.push('');}else row.push(c);}out.push(row);}return out;}
function sqName(r,c){return 'abcdefgh'[c]+(8-r);}
function draw(){const g=GAMES[gi];if(!g)return;const fen=mi===0?START:g.moves[mi-1].fen;const b=fenBoard(fen);
 const prev=mi===0?fenBoard(START):fenBoard(mi===1?START:g.moves[mi-2].fen);
 const el=document.getElementById('board');el.innerHTML='';
 for(let r=0;r<8;r++)for(let c=0;c<8;c++){const d=document.createElement('div');d.className='sq '+(((r+c)%2)?'d':'l');
  if(mi>0){if(prev[r][c]&&!b[r][c])d.classList.add('from');if(b[r][c]&&prev[r][c]!==b[r][c]&&!(prev[r][c]&&!b[r][c]))d.classList.add('to');}
  d.textContent=U[b[r][c]]||'';el.appendChild(d);}
 document.getElementById('slider').max=g.moves.length;document.getElementById('slider').value=mi;
 const m=mi>0?g.moves[mi-1]:null;
 document.getElementById('mv').textContent=mi===0?'start position':`ply ${mi}/${g.moves.length}: ${m.side} ${m.san}`;
 const info=document.getElementById('info');
 let rows=`<tr><th>result</th><td class='${g.result.includes('win')?'win':''}'>${g.result}</td></tr>`;
 if(m){rows+=`<tr><th>played</th><td>${m.san} (p ${m.p})</td></tr><tr><th>value</th><td>${m.value}</td></tr>`;
  m.top.forEach((t,i)=>{rows+=`<tr><th>${i?'':'top-3'}</th><td>${t[0]} <span class='sub'>${t[1]}</span></td></tr>`;});}
 info.innerHTML=rows;
 const vs=g.moves.map(x=>x.value);const W=352,H=60;let s='';
 vs.forEach((v,i)=>{const x=2+i/Math.max(vs.length-1,1)*(W-4);const y=H/2-Math.max(-1,Math.min(1,v/2))*(H/2-4);s+=(i?' ':'')+x.toFixed(1)+','+y.toFixed(1);});
 const cx=2+Math.max(mi-1,0)/Math.max(vs.length-1,1)*(W-4);
 document.getElementById('valplot').innerHTML=`<line x1='0' y1='${H/2}' x2='${W}' y2='${H/2}' stroke='#ddd'/><polyline points='${s}' fill='none' stroke='#2f5d8a' stroke-width='1.4'/><line x1='${cx}' y1='0' x2='${cx}' y2='${H}' stroke='#c7522e'/>`;
 try{localStorage.setItem('dash_g',gi);localStorage.setItem('dash_m',mi);}catch(e){}}
function go(n){const g=GAMES[gi];mi=Math.max(0,Math.min(g.moves.length,n));draw();}
sel.onchange=()=>{gi=+sel.value;mi=0;draw();};
document.getElementById('first').onclick=()=>go(0);document.getElementById('prev').onclick=()=>go(mi-1);
document.getElementById('next').onclick=()=>go(mi+1);document.getElementById('last').onclick=()=>go(1e9);
document.getElementById('slider').oninput=e=>go(+e.target.value);
document.getElementById('play').onclick=function(){if(timer){clearInterval(timer);timer=null;this.textContent='▶ play';return;}
 this.textContent='⏸ pause';timer=setInterval(()=>{if(mi>=GAMES[gi].moves.length){clearInterval(timer);timer=null;document.getElementById('play').textContent='▶ play';return;}go(mi+1);},600);};
document.addEventListener('keydown',e=>{if(e.key==='ArrowRight')go(mi+1);if(e.key==='ArrowLeft')go(mi-1);});
try{const sg=+localStorage.getItem('dash_g'),sm=+localStorage.getItem('dash_m');if(GAMES[sg]){gi=sg;sel.value=sg;mi=Math.min(sm||0,GAMES[sg].moves.length);}}catch(e){}
if(GAMES.length)draw();else document.getElementById('board').textContent='no games yet';
</script></body></html>""")
    return "\n".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "dashboard" / "index.html")
    ap.add_argument("--checkpoints", type=int, default=3, help="most recent arms with a model to play games for")
    ap.add_argument("--games", type=int, default=2, help="games per checkpoint")
    ap.add_argument("--watch", type=int, default=0, help="regenerate every N seconds (page auto-refreshes)")
    ap.add_argument("--open", action="store_true")
    args = ap.parse_args()
    cache_path = args.out.parent / "games_cache.json"
    while True:
        results = collect_results()
        running, progress = collect_running()
        recent = [r["name"] for r in results if r["has_model"]][: args.checkpoints]
        try:
            cache = play_games([f"experiments/{n}" for n in recent], args.games, cache_path)
        except Exception as ex:                       # keep the page alive even if a model fails to load
            print("games:", ex, file=sys.stderr)
            cache = _load(cache_path) or {}
        page = render(results, running, progress, collect_matrices(), collect_outcomes(), cache, args.watch)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(page)
        print(f"wrote {args.out} ({len(results)} arms, {len(running) + len(progress)} running, {sum(len(v['games']) for v in cache.values())} games)")
        if args.open:
            subprocess.run(["open", str(args.out)], check=False)
            args.open = False
        if not args.watch:
            break
        time.sleep(args.watch)


if __name__ == "__main__":
    main()
