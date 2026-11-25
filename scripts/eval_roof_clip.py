"""How well does CLIP name roof materials? Whole-photo prompts, over the test set with hand labels."""
import json, sys
from pathlib import Path
import numpy as np, torch
from PIL import Image
sys.path.insert(0, str(Path(__file__).parent))
import material_family as mf
from matte import matte
TRUTH = {"cottage": "slate", "mc_small_house": "wood shingles", "mc_medium_house": "wood shingles",
         "mc_big_house": "wood shingles", "mc_fletcher_house": "wood shingles", "photo_red_barn": "metal",
         "photo_aframe": "metal", "photo_brick_cottage": "clay tiles", "photo_stone_farmhouse": "slate",
         "photo_thatched_cottage": "thatch", "photo_thatched_white": "thatch"}
m, proc = mf._load()
fams = list(mf.ROOF)
templates = {"part prompts (current)": None,
             "a photo of a building with a {} roof": "a photo of a building with a {} roof",
             "a house with a {} roof": "a house with a {} roof"}
words = {"slate": "dark slate tile", "wood shingles": "wooden plank", "thatch": "thatched straw", "metal": "corrugated metal", "clay tiles": "red clay tile"}
T = {t["name"]: t for t in json.load(open("data/testset.json"))}
for tname, tpl in templates.items():
    ok = 0; res = []
    for n, truth in TRUTH.items():
        img = matte(T[n]["picture"]).convert("RGB")
        if tpl is None:
            texts, owner = [], []
            for f, (pr, _) in mf.ROOF.items():
                texts += pr; owner += [f] * len(pr)
        else:
            texts = [tpl.format(words[f]) for f in fams]; owner = fams
        with torch.no_grad():
            inp = proc(text=texts, images=img, return_tensors="pt", padding=True).to(m.device)
            p = m(**inp).logits_per_image.softmax(-1)[0].cpu().numpy()
        sc = {}
        for f, v in zip(owner, p):
            sc[f] = sc.get(f, 0) + float(v)
        best = max(sc, key=sc.get); ok += best == truth; res.append(f"{n}:{best}")
    print(f"{ok}/{len(TRUTH)}  {tname}\n   " + ", ".join(res))

# ensemble: average of part prompts and the "building with a ... roof" template
ok = 0; res = []
for n, truth in TRUTH.items():
    img = matte(T[n]["picture"]).convert("RGB")
    sc = {f: 0.0 for f in fams}
    for tpl in (None, "a house with a {} roof"):
        if tpl is None:
            texts, owner = [], []
            for f, (pr, _) in mf.ROOF.items():
                texts += pr; owner += [f] * len(pr)
        else:
            texts = [tpl.format(words[f]) for f in fams]; owner = fams
        with torch.no_grad():
            inp = proc(text=texts, images=img, return_tensors="pt", padding=True).to(m.device)
            p = m(**inp).logits_per_image.softmax(-1)[0].cpu().numpy()
        for f, v in zip(owner, p):
            sc[f] += float(v) / 2
    best = max(sc, key=sc.get); ok += best == truth; res.append(f"{n}:{best}")
print(f"{ok}/{len(TRUTH)}  ensemble\n   " + ", ".join(res))
