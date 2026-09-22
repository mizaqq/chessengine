"""Live dashboard: a local HTTP server that serves the experiment state as JSON and a
single-page app (tools/dashboard/) that polls it and draws charts.

    python -m scripts.dashboard_server [--port 8765] [--checkpoints 3] [--games 2] [--open]

Endpoints
  /                    the app (tools/dashboard/index.html, app.js)
  /api/state           running tasks (driver logs, training progress.json, pretraining logs),
                       results table, win matrices, outcome notes, game list
  /api/run?arm=<a/b>   full training log of one arm (for curve overlays)
  /api/games           cached replays for the recent checkpoints
  /api/play?arm=<a/b>  POST: play one more game with that checkpoint (background)

Games are played in a background thread and cached in dashboard/games_cache.json.
Launch outside the sandbox only if you want game generation on the GPU; the CPU is fine.
"""
import argparse
import glob
import json
import re
import subprocess
import threading
import time
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse, parse_qs

from scripts.dashboard import (ROOT, EXP, CURVES, DIALS, _load, collect_results, collect_running,
                               collect_matrices, collect_outcomes)

UI = ROOT / "tools" / "dashboard"
CACHE = ROOT / "dashboard" / "games_cache.json"
_lock = threading.Lock()
_play_queue: list = []


def collect_pretraining():
    """Supervised runs: pretrain_log.json grows during the run; the model file appears at the end."""
    out = []
    for pl in EXP.glob("human-pretraining/*/pretrain_log.json"):
        d = _load(pl) or []
        evals = [e for e in d if "eval_top1" in e and "step" in e]
        final = next((e["final"] for e in d if "final" in e), None)
        done = bool(list(pl.parent.glob("model_*.pth"))) or final is not None
        out.append({"arm": f"human-pretraining/{pl.parent.name}", "done": done, "updated": pl.stat().st_mtime,
                    "steps": d[-1].get("step") if d else None, "final": final,
                    "curve": [(e["step"], e["eval_top1"]) for e in evals],
                    "train_curve": [(e["step"], e["train_top1"]) for e in d if "train_top1" in e],
                    "last": evals[-1] if evals else {}})
    out.sort(key=lambda r: -r["updated"])
    return out


def collect_elo():
    """Finished ladders (elo.json per arm) and ladders in progress (elo_progress.json)."""
    rated, running = [], []
    for ej in EXP.glob("*/*/elo.json"):
        d = _load(ej)
        if d:
            rated.append({"arm": f"{ej.parent.parent.name}/{ej.parent.name}", "elo": d["elo"], "ci95": d["ci95"], "bound": d.get("bound"),
                          "scores": d.get("scores"), "games": d.get("games"), "greedy": d.get("greedy"), "mtime": ej.stat().st_mtime})
    for pj in EXP.glob("*/*/elo_progress.json"):
        if time.time() - pj.stat().st_mtime < 3 * 3600:
            d = _load(pj)
            if d:
                running.append({"arm": f"{pj.parent.parent.name}/{pj.parent.name}", **d})
    rated.sort(key=lambda r: -(r["elo"] or 0))
    return rated, running


def state():
    results = collect_results()
    running, progress = collect_running()
    pre = collect_pretraining()
    running_pre = [p for p in pre if not p["done"] and time.time() - p["updated"] < 2 * 3600]
    cache = _load(CACHE) or {}
    games = []
    for key, entry in sorted(cache.items(), key=lambda kv: -kv[1]["played"]):
        for i, g in enumerate(entry["games"]):
            games.append({"id": f"{key}#{i}", "arm": entry["arm"], "n": i + 1, "result": g["result"], "plies": len(g["moves"])})
    matrices = collect_matrices(limit=8)
    for m in matrices:      # tuple keys -> the exact string JSON.stringify([a, b]) produces in the browser
        m["wins"] = {json.dumps([a, b], separators=(",", ":")): v for (a, b), v in m["wins"].items()}
    rated, running_elo = collect_elo()
    return {"now": time.time(), "results": results, "running": running, "progress": progress, "pretraining": pre,
            "elo": rated, "running_elo": running_elo,
            "running_pretraining": running_pre, "matrices": matrices, "outcomes": collect_outcomes(limit=8, tail=60),
            "games": games, "curves": CURVES}


def run_logs(arm):
    d = _load(EXP / arm / "run.json")
    if d:
        return {"arm": arm, "logs": d["logs"], "config": {k: d["config"].get(k) for k in ("max_updates", "init_from", "num_filters", "device")}}
    p = _load(EXP / arm / "progress.json")
    return {"arm": arm, "logs": p["logs"] if p else [], "config": {}}


def games_payload():
    cache = _load(CACHE) or {}
    out = {}
    for key, entry in cache.items():
        for i, g in enumerate(entry["games"]):
            out[f"{key}#{i}"] = {"arm": entry["arm"], "result": g["result"], "moves": g["moves"]}
    return out


def _play_one(arm):
    import torch
    from src.model.checkpoints import load_models
    from src.viz.play import play_game
    models = sorted(glob.glob(str(EXP / arm / "model_*.pth")))
    if not models:
        return
    key = models[-1]
    with _lock:
        cache = _load(CACHE) or {}
        n = len(cache.get(key, {}).get("games", []))
    w, b, oriented, _ = load_models(EXP / arm)
    torch.manual_seed(100 + n)
    g = play_game(w, b, oriented=oriented, seed=100 + n)
    rec = {"result": g.result, "moves": [{"san": m.san, "fen": m.fen_after, "side": m.side, "value": round(m.value, 3),
                                          "top": [(s, round(p, 3)) for s, p in m.top_moves], "p": round(m.prob_played, 3)} for m in g.moves]}
    with _lock:
        cache = _load(CACHE) or {}
        entry = cache.setdefault(key, {"arm": arm, "games": [], "played": 0})
        entry["games"].append(rec); entry["played"] = time.time()
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        CACHE.write_text(json.dumps(cache))


def worker(checkpoints, games):
    """Keep `games` replays for the most recent `checkpoints` arms; serve on-demand requests."""
    while True:
        try:
            if _play_queue:
                _play_one(_play_queue.pop(0)); continue
            cache = _load(CACHE) or {}
            recent = [r["name"] for r in collect_results() if r["has_model"]][:checkpoints]
            for arm in recent:
                models = sorted(glob.glob(str(EXP / arm / "model_*.pth")))
                if models and len(cache.get(models[-1], {}).get("games", [])) < games:
                    _play_one(arm); break
            else:
                time.sleep(10)
        except Exception as ex:
            print("worker:", ex, flush=True); time.sleep(10)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # quiet
        pass

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code); self.send_header("Content-Type", "application/json"); self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

    def _file(self, path, ctype):
        body = path.read_bytes()
        self.send_response(200); self.send_header("Content-Type", ctype); self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path); q = parse_qs(u.query)
        try:
            if u.path in ("/", "/index.html"):
                return self._file(UI / "index.html", "text/html; charset=utf-8")
            if u.path == "/app.js":
                return self._file(UI / "app.js", "application/javascript")
            if u.path == "/api/state":
                return self._json(state())
            if u.path == "/api/run":
                return self._json(run_logs(q.get("arm", [""])[0]))
            if u.path == "/api/games":
                return self._json(games_payload())
            if u.path == "/api/outcome":
                p = (ROOT / q.get("file", [""])[0]).resolve()
                if p.suffix == ".md" and str(p).startswith(str(EXP)):
                    return self._json({"file": q["file"][0], "text": p.read_text()})
            self._json({"error": "not found"}, 404)
        except Exception as ex:
            self._json({"error": str(ex)}, 500)

    def do_POST(self):
        u = urlparse(self.path); q = parse_qs(u.query)
        if u.path == "/api/play":
            arm = q.get("arm", [""])[0]
            if re.fullmatch(r"[\w\-./]+", arm):
                _play_queue.append(arm)
                return self._json({"queued": arm, "queue": len(_play_queue)})
        self._json({"error": "bad request"}, 400)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--checkpoints", type=int, default=3)
    ap.add_argument("--games", type=int, default=2)
    ap.add_argument("--open", action="store_true")
    args = ap.parse_args()
    threading.Thread(target=worker, args=(args.checkpoints, args.games), daemon=True).start()
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"dashboard at http://127.0.0.1:{args.port}/", flush=True)
    if args.open:
        subprocess.run(["open", f"http://127.0.0.1:{args.port}/"], check=False)
    srv.serve_forever()


if __name__ == "__main__":
    main()
