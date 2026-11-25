"""Build every test-set picture two ways and put them side by side, so a change is judged on all buildings.

usage: python scripts/run_testset.py <tag> "--flat 0" "--flat 1"     (each quoted arg = one variant's options)
Output: out/testset/<tag>.jpg  - one row per building: picture | variant 1 | variant 2 ...
"""
import json
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
PY = str(ROOT / ".venv" / "Scripts" / "python.exe")
OUT = ROOT / "out" / "testset"


def main():
    tag, variants = sys.argv[1], sys.argv[2:] or [""]
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    import os
    only = [n for n in os.environ.get("ONLY", "").split(",") if n]
    for t in json.load(open(ROOT / "data" / "testset.json")):
        if only and t["name"] not in only:
            continue
        painted = ROOT / "out" / f"{t['name']}_painted.glb"
        if not painted.exists():
            print("skip (no model yet):", t["name"])
            continue
        imgs = []
        for i, v in enumerate(variants):
            iso = OUT / f"{t['name']}_{tag}_{i}.png"
            cmd = [PY, "-W", "ignore", "scripts/build_from_model.py", str(painted), t["picture"], f"{t['name']} test",
                   "--width", str(t["width"]), "--iso", str(iso)] + v.split()
            r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
            if r.returncode or not iso.exists():
                print("FAILED", t["name"], v, r.stderr[-800:])
                imgs.append(Image.new("RGB", (480, 480), (90, 30, 30)))
            else:
                imgs.append(Image.open(iso).convert("RGB"))
        pic = Image.open(ROOT / t["picture"]).convert("RGB")
        pic.thumbnail((480, 480))
        rows.append((t["name"], pic, imgs))
        print("done", t["name"], flush=True)
    W, H = 480, 480
    sheet = Image.new("RGB", (W * (1 + len(variants)), H * len(rows) + 30), "white")
    d = ImageDraw.Draw(sheet)
    for c, v in enumerate(["picture"] + list(variants)):
        d.text((c * W + 8, 8), v or "default", fill="black")
    for r, (name, pic, imgs) in enumerate(rows):
        y = 30 + r * H
        sheet.paste(pic, ((W - pic.width) // 2, y + (H - pic.height) // 2))
        d.text((8, y + 4), name, fill="red")
        for c, im in enumerate(imgs):
            sheet.paste(im, ((c + 1) * W, y))
    f = OUT / f"{tag}.jpg"
    sheet.save(f, quality=88)
    print("sheet:", f)


if __name__ == "__main__":
    main()
