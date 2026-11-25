"""Side-by-side sheet of two test-set runs: python compare_sheet.py <tagA> <tagB> <out.jpg> [labelA] [labelB]"""
import json, sys
from pathlib import Path
from PIL import Image, ImageDraw
ROOT = Path(__file__).resolve().parent.parent
ta, tb, out = sys.argv[1:4]
la = sys.argv[4] if len(sys.argv) > 4 else ta
lb = sys.argv[5] if len(sys.argv) > 5 else tb
W = 300
rows = []
for t in json.load(open(ROOT / "data" / "testset.json")):
    a, b = ROOT / f"out/testset/{t['name']}_{ta}_0.png", ROOT / f"out/testset/{t['name']}_{tb}_0.png"
    if a.exists() and b.exists():
        pic = Image.open(ROOT / t["picture"]).convert("RGB"); pic.thumbnail((W, W))
        rows.append((t["name"], pic, Image.open(a).resize((W, W)), Image.open(b).resize((W, W))))
cols = 2
s = Image.new("RGB", (W * 3 * cols, W * ((len(rows) + 1) // cols)), "white"); d = ImageDraw.Draw(s)
for i, (n, pic, a, b) in enumerate(rows):
    x0, y = (i % cols) * W * 3, (i // cols) * W
    s.paste(pic, (x0 + (W - pic.width) // 2, y + (W - pic.height) // 2)); s.paste(a, (x0 + W, y)); s.paste(b, (x0 + 2 * W, y))
    d.text((x0 + 4, y + 4), n, fill="red"); d.text((x0 + W + 4, y + 4), la, fill="red"); d.text((x0 + 2 * W + 4, y + 4), lb, fill="red")
s.save(ROOT / out, quality=85); print(len(rows), "rows ->", out)
