"""Shared HTTP fetching with on-disk caching and polite rate limiting."""

import hashlib
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW = DATA / "raw"

UA = "american-dialects-research/1.0 (personal research project)"
DELAY = 0.4

_last_request = {"t": 0.0}


def fetch(url, cache_dir, filename=None, binary=False, referer=None, force=False):
    """Fetch a URL, caching the response under data/raw/<cache_dir>/."""
    d = RAW / cache_dir
    d.mkdir(parents=True, exist_ok=True)
    if filename is None:
        filename = hashlib.sha1(url.encode()).hexdigest()[:16]
    path = d / filename

    if path.exists() and path.stat().st_size > 0 and not force:
        return path.read_bytes() if binary else path.read_text(encoding="utf-8", errors="replace")

    wait = DELAY - (time.time() - _last_request["t"])
    if wait > 0:
        time.sleep(wait)

    headers = {"User-Agent": UA}
    if referer:
        headers["Referer"] = referer

    last_err = None
    for attempt in range(4):
        try:
            r = requests.get(url, headers=headers, timeout=45)
            _last_request["t"] = time.time()
            r.raise_for_status()
            path.write_bytes(r.content)
            return r.content if binary else r.content.decode("utf-8", errors="replace")
        except Exception as e:
            last_err = e
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"failed to fetch {url}: {last_err}")


def out_dir(name):
    d = DATA / name
    d.mkdir(parents=True, exist_ok=True)
    return d
