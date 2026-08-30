"""A local site for the party trick.

Loads the model once, then serves an adaptive quiz over plain HTTP. No
framework and no new dependencies: the standard library serves the files,
Pillow draws the map, and the model is the same one quiz.py drives from the
terminal.

Every finished game is appended to data/quiz/log.csv in exactly the format
calibrate.py --set quiz expects, so playing the web version feeds the one
measurement the report says is still missing.

    ../.venv/bin/python server.py        then open http://localhost:8000
"""

import csv
import io
import json
import secrets
import sys
import threading
import webbrowser
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "model"))
sys.path.insert(0, str(ROOT.parent / "scrape"))

from choose import Selector            # noqa: E402
from infer import N_QUESTIONS, Geolocator, Places   # noqa: E402
from common import DATA                # noqa: E402

STATIC = ROOT / "static"
LOG = DATA / "quiz" / "log.csv"
LOG_HEADER = ["played_at", "person", "truth", "lat", "lon", "n_asked",
              "n_answered", "tau", "map_lat", "map_lon", "map_state",
              "top_place", "answers"]

MODEL = {}
SESSIONS = {}
LOCK = threading.Lock()


# --------------------------------------------------------------------------
# the map
# --------------------------------------------------------------------------

LAND = (232, 232, 237)
RAMP = [(0.00, (222, 233, 246)),
        (0.35, (138, 180, 233)),
        (0.70, (0, 113, 227)),
        (1.00, (0, 40, 112))]


def _lut(n=256):
    """A smooth single-hue ramp, land colour at the bottom end."""
    out = np.zeros((n, 3), dtype=np.float64)
    xs = np.linspace(0, 1, n)
    for i, x in enumerate(xs):
        for (a, ca), (b, cb) in zip(RAMP, RAMP[1:]):
            if a <= x <= b:
                f = 0.0 if b == a else (x - a) / (b - a)
                out[i] = [ca[j] + f * (cb[j] - ca[j]) for j in range(3)]
                break
    return out


LUT = _lut()


def render(post, t, scale=3, gamma=0.4):
    """The posterior as a small PNG: grey land, blue where the mass is.

    The raw posterior is far too peaked to look at directly -- one cell can
    hold a thousand times another and the map reads as a single dot on an
    empty country. Dividing by the maximum and raising to a fractional power
    lifts the shoulders back into view without changing the ordering, so the
    picture shows the shape of the belief rather than only its argmax.
    """
    g = t.grid(post)
    land = ~np.isnan(g)
    v = np.where(land, np.nan_to_num(g), 0.0)
    top = v.max()
    v = (v / top) ** gamma if top > 0 else v

    idx = np.clip((v * (len(LUT) - 1)).astype(int), 0, len(LUT) - 1)
    rgb = LUT[idx]
    rgb[~land] = 255
    rgb[land & (v <= 0)] = LAND

    img = Image.fromarray(rgb.astype(np.uint8), mode="RGB")
    alpha = Image.fromarray((land * 255).astype(np.uint8), mode="L")
    img.putalpha(alpha)
    if scale > 1:
        img = img.resize((img.width * scale, img.height * scale), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


# --------------------------------------------------------------------------
# the quiz
# --------------------------------------------------------------------------

def load_text():
    qs, ans = {}, {}
    with open(DATA / "hds" / "questions.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            qs[r["question"]] = r["text"]
    with open(DATA / "hds" / "answers.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            ans.setdefault(r["question"], {})[r["choice"]] = r["answer"]
    return qs, ans


def boot():
    g = Geolocator()
    MODEL["g"] = g
    MODEL["places"] = Places(min_pop=20000)
    MODEL["sel"] = Selector(g)
    MODEL["qs"], MODEL["ans"] = load_text()


def next_question(s):
    """The question that would tell us most about *this* person right now."""
    g, sel = MODEL["g"], MODEL["sel"]
    remaining = [q for q in g.t.questions if q not in s["asked"]]
    if not remaining:
        return None
    info = sel.information(s["w"][None, :], remaining)
    q = max(remaining, key=lambda x: float(info[x].mean()))
    choices = [{"id": g.t.choice[i],
                "text": MODEL["ans"].get(q, {}).get(g.t.choice[i], g.t.choice[i])}
               for i in g.t.rows[q]]
    return {"id": q, "text": MODEL["qs"].get(q, f"question {q}"),
            "choices": choices}


def guess(s):
    """What the model would say out loud, given what it has been told."""
    g, places = MODEL["g"], MODEL["places"]
    if not s["answers"]:
        return None
    post = g.posterior(s["answers"])
    best = int(np.argmax(post))
    areas = places.areas(g.t, post, 3)
    states = sorted(((str(st), float(post[g.t.state == st].sum()))
                     for st in np.unique(g.t.state)), key=lambda kv: -kv[1])[:3]
    order = np.argsort(post)[::-1]
    n80 = int(np.searchsorted(np.cumsum(post[order]), 0.8) + 1)
    km2 = float(g.cell_km2[order[:n80]].sum())
    s["post"] = post
    return {
        "places": [{"name": n, "state": st, "p": float(p)} for n, st, p in areas],
        "states": [{"state": st, "p": p} for st, p in states],
        "lat": float(g.t.cell_lat[best]), "lon": float(g.t.cell_lon[best]),
        "state": str(g.t.state[best]),
        "km2": km2,
        "tau": float(g.tau_used(s["answers"])),
        "answered": len(s["answers"]),
    }


def log_game(s, truth, lat, lon):
    g = MODEL["g"]
    if "post" not in s or not s["answers"]:
        return
    best = int(np.argmax(s["post"]))
    areas = MODEL["places"].areas(g.t, s["post"], 1)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    new = not LOG.exists()
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with open(LOG, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(LOG_HEADER)
        w.writerow([now, f"w{abs(hash(now)) % 10**8:08d}", truth, lat, lon,
                    len(s["asked"]), len(s["answers"]),
                    f"{g.tau_used(s['answers']):.3f}",
                    f"{g.t.cell_lat[best]:.4f}", f"{g.t.cell_lon[best]:.4f}",
                    g.t.state[best], areas[0][0] if areas else "",
                    ";".join(f"{q}:{c}" for q, c in s["answers"])])


# --------------------------------------------------------------------------
# http
# --------------------------------------------------------------------------

TYPES = {".html": "text/html; charset=utf-8", ".css": "text/css; charset=utf-8",
         ".js": "text/javascript; charset=utf-8", ".png": "image/png",
         ".svg": "image/svg+xml"}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json", cache=False):
        if isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control",
                         "public, max-age=3600" if cache else "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj), "application/json")

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(n) or b"{}")

    def _session(self, sid):
        with LOCK:
            return SESSIONS.get(sid)

    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/":
            path = "/index.html"

        if path.startswith("/api/map/"):
            sid = path[len("/api/map/"):].removesuffix(".png")
            s = self._session(sid)
            if not s or "post" not in s:
                return self._send(404, b"", "image/png")
            return self._send(200, render(s["post"], MODEL["g"].t), "image/png")

        f = (STATIC / path.lstrip("/")).resolve()
        if not str(f).startswith(str(STATIC)) or not f.is_file():
            return self._send(404, "not found", "text/plain; charset=utf-8")
        return self._send(200, f.read_bytes(),
                          TYPES.get(f.suffix, "application/octet-stream"))

    def do_POST(self):
        path = self.path.split("?")[0]
        try:
            data = self._body()
        except Exception:
            return self._json({"error": "bad json"}, 400)

        if path == "/api/start":
            sid = secrets.token_urlsafe(12)
            s = {"asked": [], "answers": [],
                 "w": MODEL["g"].prior.astype(np.float64).copy(),
                 "n": int(data.get("n") or N_QUESTIONS)}
            with LOCK:
                SESSIONS[sid] = s
            return self._json({"session": sid, "n": s["n"],
                               "question": next_question(s)})

        s = self._session(data.get("session", ""))
        if s is None:
            return self._json({"error": "no such session"}, 404)

        if path == "/api/answer":
            g = MODEL["g"]
            q, choice = str(data.get("question")), data.get("choice")
            if q not in s["asked"]:
                s["asked"].append(q)
            if choice:
                s["answers"].append((q, str(choice)))
                i = g.index[(q, str(choice))]
                ll = g.loglik(i)
                s["w"] = s["w"] * np.exp(ll - ll.max())
                total = s["w"].sum()
                if total > 0:
                    s["w"] /= total
            done = len(s["asked"]) >= s["n"]
            return self._json({
                "done": done,
                "asked": len(s["asked"]),
                "question": None if done else next_question(s),
                "guess": guess(s),
            })

        if path == "/api/finish":
            return self._json({"done": True, "asked": len(s["asked"]),
                               "guess": guess(s)})

        if path == "/api/log":
            log_game(s, str(data.get("truth", "")).strip(),
                     str(data.get("lat", "")).strip(),
                     str(data.get("lon", "")).strip())
            return self._json({"ok": True, "path": str(LOG)})

        return self._json({"error": "not found"}, 404)


def main():
    port = int(sys.argv[sys.argv.index("--port") + 1]) if "--port" in sys.argv else 8000
    print("loading the model ...", flush=True)
    boot()
    url = f"http://localhost:{port}"
    print(f"ready -> {url}\nctrl-c to stop")
    if "--no-open" not in sys.argv:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
