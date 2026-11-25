"""Texture variety: break up a flat wall of one block with related blocks, the way builders do.

Which blocks go with which comes from builders themselves: 3000 top-liked palettes from blockpalettes.com
(data/blockpalettes_top3000.json). A variant must (1) often appear in palettes together with the main block,
(2) look close in colour, (3) be a full block. The mix is patchy (3D noise), not salt-and-pepper, and mossy /
cracked variants lean toward the ground, like real weathering.
"""
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from convert import LEAF_TINT, TEX, to_lab

DATA = Path(__file__).resolve().parent.parent / "data" / "blockpalettes_top3000.json"
NOT_FULL = ("glass", "slab", "stairs", "wall", "fence", "door", "pane", "carpet", "leaves", "lantern", "chain",
            "bars", "button", "plate", "sign", "banner", "candle", "torch", "rod", "ladder", "rail", "sapling",
            "flower", "head", "skull", "bed", "pot", "scaffolding", "cobweb", "vine", "grass", "fern", "bush",
            "lichen", "froglight", "barrel", "observer", "piston", "dispenser", "dropper", "furnace", "table",
            "lectern", "composter", "beehive", "bee_nest", "shulker", "chest", "anvil", "bell", "campfire",
            "cauldron", "hopper", "grindstone", "stonecutter", "loom", "smoker", "jukebox", "note_block",
            "sculk", "tnt", "spawner", "pumpkin", "melon", "sponge", "slime", "honey", "ice", "snow", "powder",
            "sand", "gravel", "concrete_powder", "mushroom", "wart", "magma", "glowstone", "sea_lantern",
            "redstone", "lamp", "ore", "amethyst_cluster", "bud", "pointed", "coral", "kelp", "frame", "crafter",
            "trial", "vault", "heavy_core", "copper_bulb", "copper_grate", "end_rod", "crying", "respawn",
            "lodestone", "beacon", "conduit", "enchanting", "brewing", "dragon", "command", "structure", "jigsaw",
            "barrier", "bookshelf", "target", "scaffold", "nylium", "mycelium", "podzol", "farmland",
            "path", "dirt", "clay", "soul", "bone", "hay", "dried_kelp", "pearlescent", "verdant", "ochre")
WEATHER = ("mossy", "cracked")

_tex_cache = {}


def tex_lab(name):
    """Mean Lab of a block's texture (side texture when it has one), or None if no texture."""
    if name in _tex_cache:
        return _tex_cache[name]
    lab = None
    alias = {"smooth_sandstone": "sandstone_top", "smooth_red_sandstone": "red_sandstone_top",
             "smooth_quartz": "quartz_block_bottom", "smooth_stone": "smooth_stone"}
    for cand in ([alias[name]] if name in alias else []) + [name, name + "_side", name.replace("_block", ""),
                                                               name + "_top", name.replace("bricks", "brick")]:
        f = TEX / f"{cand}.png"
        if f.exists():
            px = np.asarray(Image.open(f).convert("RGBA")).astype(np.float64)[:16]
            rgb = px[..., :3][px[..., 3] > 0]
            if "leaves" in name:
                rgb = rgb * LEAF_TINT / 0.55
            lab = to_lab(np.clip(rgb, 0, 255)).mean(0)
            break
    _tex_cache[name] = lab
    return lab


_pairs = None


def _cooccurrence():
    global _pairs
    if _pairs is None:
        pal = json.load(open(DATA))
        single, pair = Counter(), Counter()
        for p in pal:
            bs = set(p["blocks"])
            w = 1 + np.log1p(p["likes"])
            for a in bs:
                single[a] += w
                for b in bs:
                    if a != b:
                        pair[(a, b)] += w
        _pairs = (single, pair)
    return _pairs


WOODY = ("planks", "log", "wood", "stem", "hyphae", "bamboo")
SIBLINGS = {  # a block's own family members, always allowed when close in colour
    "sandstone": ["smooth_sandstone", "cut_sandstone", "chiseled_sandstone"],
    "red_sandstone": ["smooth_red_sandstone", "cut_red_sandstone", "chiseled_red_sandstone"],
    "stone_bricks": ["mossy_stone_bricks", "cracked_stone_bricks", "chiseled_stone_bricks"],
    "deepslate_bricks": ["cracked_deepslate_bricks", "deepslate_tiles", "cracked_deepslate_tiles"],
    "deepslate_tiles": ["cracked_deepslate_tiles", "deepslate_bricks"],
    "nether_bricks": ["cracked_nether_bricks", "chiseled_nether_bricks"],
    "polished_blackstone_bricks": ["cracked_polished_blackstone_bricks"],
    "quartz_block": ["quartz_bricks", "smooth_quartz", "chiseled_quartz_block"],
    "end_stone_bricks": ["end_stone"],
}


def _full(n):
    """Whole-word check, so 'sand' doesn't catch sandstone and 'mud' doesn't catch mud_bricks."""
    words = set(n.split("_"))
    return not any((t in words) if "_" not in t else (t in n) for t in NOT_FULL)


def _woody(n):
    return any(t in n for t in WOODY)


def variants(main, max_n=3, max_de=16.0):
    """[(block, weight)] that builders pair with `main` and that look close to it."""
    single, pair = _cooccurrence()
    m = tex_lab(main)
    if m is None or single[main] == 0:
        return []
    out = []
    cands = {b: w for (a, b), w in pair.items() if a == main}
    for b in SIBLINGS.get(main, []):
        cands[b] = max(cands.get(b, 0), 0.15 * single[main])
    for b, w in cands.items():
        if not _full(b) or _woody(b) != _woody(main):
            continue
        lab = tex_lab(b)
        if lab is None:
            continue
        de = float(np.sqrt(((lab - m) ** 2).sum()))
        if de > max_de:
            continue
        lift = w / single[main]  # how often a palette with `main` also has b
        if lift < 0.04:
            continue
        out.append((b, lift * (1 - de / (max_de * 1.5))))
    out.sort(key=lambda t: -t[1])
    return out[:max_n]


def _noise(p, scale, seed):
    """Smooth 3D value noise in [0, 1] (patches a few blocks across)."""
    x, y, z = (v / scale for v in p)
    xi, yi, zi = int(np.floor(x)), int(np.floor(y)), int(np.floor(z))
    fx, fy, fz = x - xi, y - yi, z - zi

    def h(i, j, k):
        n = (i * 73856093) ^ (j * 19349663) ^ (k * 83492791) ^ (seed * 2654435761)
        n = (n ^ (n >> 13)) * 1274126177
        return ((n ^ (n >> 16)) & 0xFFFF) / 65535.0

    def lerp(a, b, t):
        t = t * t * (3 - 2 * t)
        return a + (b - a) * t

    c = [[[h(xi + i, yi + j, zi + k) for k in (0, 1)] for j in (0, 1)] for i in (0, 1)]
    return lerp(lerp(lerp(c[0][0][0], c[0][0][1], fz), lerp(c[0][1][0], c[0][1][1], fz), fy),
                lerp(lerp(c[1][0][0], c[1][0][1], fz), lerp(c[1][1][0], c[1][1][1], fz), fy), fx)


def apply(mat, keys, amount=0.35, log=print):
    """Replace part of each main block with its variants. mat: {pos: block name}; returns new mat."""
    counts = Counter(n for n in mat.values() if not n.startswith("_") and "leaves" not in n)
    ymin = min(p[1] for p in keys)
    ymax = max(p[1] for p in keys)
    out = dict(mat)
    for main, n in counts.most_common():
        if n < 30:
            continue
        vs = variants(main)
        if not vs:
            continue
        log(f"  variety for {main}: " + ", ".join(f"{b}" for b, _ in vs))
        tot = sum(w for _, w in vs)
        for p in keys:
            if mat[p] != main:
                continue
            r = _noise(p, 3.0, 1)
            if r > amount:
                continue
            # pick the variant by a second noise; weathered variants more likely low down
            low = 1 - (p[1] - ymin) / max(1, ymax - ymin)
            ws = [w * (1 + 1.5 * low if any(t in b for t in WEATHER) else 1) for b, w in vs]
            t = _noise(p, 2.0, 7) * sum(ws)
            acc = 0
            for (b, _), w in zip(vs, ws):
                acc += w
                if t <= acc:
                    out[p] = b
                    break
    return out
