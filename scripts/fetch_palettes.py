"""Download the most-liked builder palettes from blockpalettes.com (used for texture variety and trim).

usage: python scripts/fetch_palettes.py [--count 3000]
Saves data/blockpalettes_top<count>.json. The data isn't shipped with this repo; it's fetched from the site.
Runs automatically the first time the pipeline needs it.
"""
import argparse
import json
import time
import urllib.request
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"
API = "https://www.blockpalettes.com/api/palettes/all_palettes.php?sort=popular&page={page}&limit=100"
KEYS = ("blockOne", "blockTwo", "blockThree", "blockFour", "blockFive", "blockSix")


def fetch(count=3000, out=None, log=print):
    out = Path(out) if out else DATA / f"blockpalettes_top{count}.json"
    palettes = []
    for page in range(1, (count + 99) // 100 + 1):
        req = urllib.request.Request(API.format(page=page), headers={"User-Agent": "build-studio palette fetch"})
        d = json.load(urllib.request.urlopen(req, timeout=30))
        for p in d.get("palettes", []):
            palettes.append({"likes": p["likes"], "blocks": [p[k] for k in KEYS if p.get(k)]})
        if page % 10 == 0:
            log(f"  {len(palettes)} palettes so far")
        if not d.get("palettes"):
            break
        time.sleep(1.0)  # be polite to the site
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(palettes[:count]))
    log(f"saved {len(palettes[:count])} palettes to {out}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=3000)
    fetch(ap.parse_args().count)
