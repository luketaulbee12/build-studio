"""Load real Minecraft builds from many file formats into one format, for learning how builders build.

A build = (palette: list of block-state strings, blocks: int array [x, y, z] of palette indices, -1 = air).
Formats: vanilla structure .nbt, Litematica .litematic, Sponge .schem (v2/v3), and the JSON/Parquet grids
some datasets use. `python scripts/corpus.py` converts everything under corpus/raw into corpus/builds/*.npz
and prints statistics.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "studio"))
from studio import read_nbt, state_string  # noqa: E402

RAW = ROOT / "corpus" / "raw"
OUT = ROOT / "corpus" / "builds"
AIR = ("minecraft:air", "minecraft:cave_air", "minecraft:void_air", "minecraft:structure_void")
MAX_SIDE = 160  # skip map-sized dumps


def _grid(size, palette, items):
    """items: iterable of (x, y, z, palette index)."""
    g = np.full(size, -1, np.int32)
    for x, y, z, i in items:
        g[x, y, z] = i
    return _clean(palette, g)


def _clean(palette, g):
    """Drop air from the palette, crop to the occupied box."""
    keep = [i for i, s in enumerate(palette) if s.split("[")[0] not in AIR]
    remap = np.full(len(palette) + 1, -1, np.int32)
    for new, old in enumerate(keep):
        remap[old] = new
    g = remap[g]  # -1 maps to remap[-1] = -1
    occ = np.argwhere(g >= 0)
    if len(occ) == 0:
        return None
    lo, hi = occ.min(0), occ.max(0) + 1
    return [palette[i] for i in keep], g[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]


def load_structure_nbt(path):
    root = read_nbt(path.read_bytes())
    pal = root.get("palette") or (root.get("palettes") or [[]])[0]
    palette = [state_string(p) for p in pal]
    size = tuple(int(v) for v in root["size"])
    return _grid(size, palette, ((b["pos"][0], b["pos"][1], b["pos"][2], b["state"]) for b in root.get("blocks", [])))


def _unpack_tight(longs, bits, count):
    """Litematica's bit array: entries run continuously across 64-bit words."""
    words = np.asarray(longs, dtype=np.int64).view(np.uint64)
    idx = np.arange(count, dtype=np.uint64) * np.uint64(bits)
    w = (idx >> np.uint64(6)).astype(np.int64)
    off = idx & np.uint64(63)
    mask = np.uint64((1 << bits) - 1)
    lo = words[w] >> off
    spill = off + np.uint64(bits) > np.uint64(64)
    hi_idx = np.minimum(w + 1, len(words) - 1)
    hi = np.where(spill, words[hi_idx] << (np.uint64(64) - off), np.uint64(0))
    return ((lo | hi) & mask).astype(np.int64)


def load_litematic(path):
    root = read_nbt(path.read_bytes())
    best = None
    for reg in root.get("Regions", {}).values():
        sz = reg["Size"]
        sx, sy, sz_ = abs(int(sz["x"])), abs(int(sz["y"])), abs(int(sz["z"]))
        palette = [state_string(p) for p in reg["BlockStatePalette"]]
        bits = max(2, math.ceil(math.log2(max(2, len(palette)))))
        n = sx * sy * sz_
        idx = _unpack_tight(reg["BlockStates"], bits, n)
        g = idx.reshape(sy, sz_, sx).transpose(2, 0, 1)  # litematica order: index = (y * sz + z) * sx + x
        g = np.where(g < len(palette), g, 0).astype(np.int32)
        r = _clean(palette, g)
        if r and (best is None or r[1].size > best[1].size):
            best = r  # keep the biggest region
    return best


def _varints(data, count):
    out = np.empty(count, np.int64)
    i = j = 0
    while j < count and i < len(data):
        v = s = 0
        while True:
            b = data[i]; i += 1
            v |= (b & 0x7F) << s
            if not b & 0x80:
                break
            s += 7
        out[j] = v; j += 1
    return out[:j]


def load_schem(path):
    root = read_nbt(path.read_bytes())
    root = root.get("Schematic", root)
    blocks = root.get("Blocks", root)  # v3 keeps palette+data under Blocks
    pal = blocks.get("Palette") or root.get("Palette")
    data = blocks.get("Data") if "Data" in blocks else root.get("BlockData")
    if pal is None or data is None:
        return None
    w, h, l = int(root["Width"]), int(root["Height"]), int(root["Length"])
    palette = [None] * (max(pal.values()) + 1)
    for k, v in pal.items():
        palette[v] = k
    palette = [p or "minecraft:air" for p in palette]
    idx = _varints(bytes(b & 0xFF for b in data) if isinstance(data, list) else bytes(data), w * h * l)
    if len(idx) < w * h * l:
        return None
    g = idx.reshape(h, l, w).transpose(2, 0, 1).astype(np.int32)  # index = (y * l + z) * w + x
    return _clean(palette, g)


def load_grid_json(path):
    d = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    blocks = d.get("blocks") if isinstance(d, dict) else None
    if not blocks:
        return None
    pal, index, items = [], {}, []
    for b in blocks:
        name = b.get("block") or b.get("name") or b.get("type")
        if not name:
            continue
        name = name if ":" in name else "minecraft:" + name
        if name not in index:
            index[name] = len(pal); pal.append(name)
        items.append((int(b["x"]), int(b["y"]), int(b["z"]), index[name]))
    if not items:
        return None
    a = np.array([i[:3] for i in items]); a -= a.min(0)
    size = tuple(a.max(0) + 1)
    return _grid(size, pal, ((x, y, z, i[3]) for (x, y, z), i in zip(a, items)))


def load_craftassist(path):
    """CraftAssist house: schematic.npy in (y, z, x, id/meta), 1.12 numeric ids."""
    from legacy_blocks import to_state
    a = np.load(path)
    ids, metas = a[..., 0], a[..., 1]
    pal, index = [], {}
    g = np.full(ids.shape, -1, np.int32)
    for (bid, meta) in {(int(i), int(m)) for i, m in zip(ids.ravel(), metas.ravel()) if i}:
        st = to_state(bid, meta)
        if st is None:
            continue
        if st not in index:
            index[st] = len(pal); pal.append(st)
        g[(ids == bid) & (metas == meta)] = index[st]
    if not pal:
        return None
    return _clean(pal, g.transpose(2, 0, 1))  # (y, z, x) -> (x, y, z)


LOADERS = {".nbt": load_structure_nbt, ".litematic": load_litematic, ".schem": load_schem}


def convert_all(log=print):
    OUT.mkdir(parents=True, exist_ok=True)
    stats = {"ok": 0, "skipped": 0, "failed": 0}
    for f in sorted(RAW.rglob("*")):
        if not f.is_file():
            continue
        loader = LOADERS.get(f.suffix)
        if f.name == "schematic.npy":
            loader = load_craftassist
        if loader is None:
            continue
        name = (f.relative_to(RAW).as_posix().replace("/", "__"))[:150]
        dst = OUT / (name + ".npz")
        if dst.exists():
            stats["ok"] += 1
            continue
        try:
            r = loader(f)
        except Exception as e:  # odd/old files: count and move on
            stats["failed"] += 1
            continue
        if r is None or max(r[1].shape) > MAX_SIDE or (r[1] >= 0).sum() < 20:
            stats["skipped"] += 1
            continue
        np.savez_compressed(dst, blocks=r[1], palette=np.array(r[0]), source=str(f.relative_to(RAW)))
        stats["ok"] += 1
    log(stats)
    return stats


if __name__ == "__main__":
    convert_all()
    sizes = []
    for f in OUT.glob("*.npz"):
        d = np.load(f)
        sizes.append(((d["blocks"] >= 0).sum(), d["blocks"].shape))
    print(f"{len(sizes)} builds, {sum(s for s, _ in sizes):,} blocks total; median build "
          f"{int(np.median([s for s, _ in sizes])):,} blocks")
