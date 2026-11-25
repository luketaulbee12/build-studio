"""Learn how real builders finish buildings, from the corpus (corpus/builds/*.npz).

For every real building (has a door or windows, sensible size) each block gets a role from its geometry:
  wall   - full block with exactly one side open to the outside air
  corner - full block where two walls meet (two outside sides at right angles), in a vertical run >= 3
  base   - wall/corner blocks in the lowest wall layer
  frame  - wall-plane blocks touching a window (glass / pane), left/right/above/below
  roof   - stairs/slabs with outside air above them
Then, per building, the main wall block and what was used for each role. Aggregated over all buildings:
  P(role block | wall block), and how often builders use a *different* block for a role at all.
Saves data/learned_roles.json. Run: python scripts/mine_patterns.py
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy import ndimage

ROOT = Path(__file__).resolve().parent.parent
BUILDS = ROOT / "corpus" / "builds"
OUT = ROOT / "data" / "learned_roles.json"

NOT_FULL = ("stairs", "slab", "fence", "wall", "pane", "door", "trapdoor", "torch", "lantern", "carpet", "button",
            "pressure_plate", "sign", "banner", "ladder", "vine", "flower", "grass", "fern", "sapling", "leaves",
            "glass", "bars", "chain", "rail", "bed", "chest", "barrel", "pot", "candle", "head", "skull", "snow",
            "water", "lava", "web", "rod", "lever", "repeater", "comparator", "redstone_wire", "campfire", "cake",
            "bell", "anvil", "cauldron", "hopper", "lectern", "composter", "table", "furnace", "smoker", "loom",
            "stonecutter", "grindstone", "scaffolding", "dripleaf", "azalea", "bush", "mushroom", "kelp", "coral",
            "sea_pickle", "lily", "cactus", "bamboo", "sugar_cane", "wheat", "carrots", "potatoes", "beetroots",
            "farmland", "dirt_path", "air", "light", "frame", "painting", "pickle", "amethyst_cluster", "bud")
GROUND = ("grass_block", "dirt", "coarse_dirt", "podzol", "sand", "gravel", "stone", "deepslate", "water", "mud",
          "rooted_dirt", "moss_block", "snow_block", "netherrack", "end_stone", "red_sand", "clay")
DIRS = ((1, 0, 0), (-1, 0, 0), (0, 0, 1), (0, 0, -1))


def base_name(state):
    n = state.split("[")[0].replace("minecraft:", "")
    return n


def is_full(n):
    return not any(t in n for t in NOT_FULL)


def roles(blocks, palette):
    """Role sets for one build. blocks: int [x, y, z] (-1 air)."""
    names = [base_name(s) for s in palette]
    full_ids = np.array([is_full(n) for n in names] + [False])
    glass_ids = np.array([("glass" in n) or n == "iron_bars" for n in names] + [False])
    solid = blocks >= 0
    # outside air: flood fill from the padded box boundary
    pad = np.pad(~solid, 1, constant_values=True)
    lab, _ = ndimage.label(pad)
    outside = (lab == lab[0, 0, 0])[1:-1, 1:-1, 1:-1]
    outside_p = np.pad(outside, 1, constant_values=True)
    full = full_ids[np.where(blocks >= 0, blocks, len(names))]
    glass = glass_ids[np.where(blocks >= 0, blocks, len(names))]
    X, Y, Z = blocks.shape
    open_sides = []
    for dx, _, dz in DIRS:
        open_sides.append(outside_p[1 + dx:1 + dx + X, 1:1 + Y, 1 + dz:1 + dz + Z])
    o = np.stack(open_sides)  # [4, X, Y, Z]: +x, -x, +z, -z
    n_open = o.sum(0)
    wall = full & (n_open == 1)
    corner_c = full & (n_open == 2) & ~((o[0] & o[1]) | (o[2] & o[3]))
    # vertical runs >= 3
    run = corner_c.astype(int)
    up = np.zeros_like(run); dn = np.zeros_like(run)
    for y in range(1, Y):
        up[:, y] = np.where(corner_c[:, y], up[:, y - 1] + 1, 0)
    for y in range(Y - 2, -1, -1):
        dn[:, y] = np.where(corner_c[:, y], dn[:, y + 1] + 1, 0)
    corner = corner_c & (up + dn - 1 >= 3)
    # base: lowest layer that has walls
    walls_any = (wall | corner).any((0, 2))
    ylow = int(np.argmax(walls_any)) if walls_any.any() else 0
    base = (wall | corner) & (np.arange(Y)[None, :, None] == ylow)
    # frame: wall-plane neighbours of glass (horizontal + vertical)
    near = np.zeros_like(glass)
    for dx, dy, dz in ((1, 0, 0), (-1, 0, 0), (0, 0, 1), (0, 0, -1), (0, 1, 0), (0, -1, 0)):
        sh = np.roll(glass, (dx, dy, dz), (0, 1, 2))
        near |= sh
    frame = near & full & (n_open >= 1)
    # roof: stairs/slabs with outside above
    stair_ids = np.array([("stairs" in n or "slab" in n) for n in names] + [False])
    stair = stair_ids[np.where(blocks >= 0, blocks, len(names))]
    above_out = np.concatenate([outside[:, 1:], np.ones((X, 1, Z), bool)], 1)
    roof = stair & above_out & (np.arange(Y)[None, :, None] > ylow + 1)
    return names, {"wall": wall & ~corner & ~frame & ~base, "corner": corner & ~base, "base": base,
                   "frame": frame & ~corner, "roof": roof}


def is_building(blocks, names):
    flat = [names[i] for i in np.unique(blocks[blocks >= 0])]
    has_door = any(n.endswith("_door") for n in flat)
    has_glass = any("glass" in n for n in flat)
    X, Y, Z = blocks.shape
    return (has_door or has_glass) and 4 <= Y <= 64 and 5 <= max(X, Z) <= 80


def norm(n):
    """Treat log/wood axes and stripped variants as the block family a builder picks."""
    return n


def mine(log=print):
    per_wall = defaultdict(lambda: {"n": 0, "corner": Counter(), "frame": Counter(), "base": Counter(),
                                    "roof": Counter(), "differs": Counter()})
    n_buildings = 0
    for f in sorted(BUILDS.glob("*.npz")):
        d = np.load(f)
        blocks, palette = d["blocks"], [str(s) for s in d["palette"]]
        names = [base_name(s) for s in palette]
        if not is_building(blocks, names):
            continue
        names, R = roles(blocks, palette)
        if R["wall"].sum() < 15:
            continue
        def top(mask, skip_ground=True):
            ids = blocks[mask]
            c = Counter(names[i] for i in ids[ids >= 0])
            for g in (GROUND if skip_ground else ()):
                c.pop(g, None)
            return c
        wall_c = top(R["wall"])
        if not wall_c:
            continue
        wall_block = wall_c.most_common(1)[0][0]
        e = per_wall[wall_block]
        e["n"] += 1
        n_buildings += 1
        for role in ("corner", "frame", "base", "roof"):
            c = top(R[role])
            if sum(c.values()) < 3:
                continue
            blk, cnt = c.most_common(1)[0]
            if cnt < 0.4 * sum(c.values()):
                continue  # no clear choice for this role in this building
            e[role][blk] += 1
            if blk != wall_block:
                e["differs"][role] += 1
    out = {"buildings": n_buildings, "walls": {}}
    for w, e in sorted(per_wall.items(), key=lambda t: -t[1]["n"]):
        if e["n"] < 3:
            continue
        out["walls"][w] = {"n": e["n"]}
        for role in ("corner", "frame", "base", "roof"):
            tot = sum(e[role].values())
            if tot:
                out["walls"][w][role] = {"choices": [[b, round(c / tot, 3)] for b, c in e[role].most_common(6)],
                                         "seen": tot, "differs": round(e["differs"][role] / tot, 3)}
    OUT.write_text(json.dumps(out, indent=1))
    log(f"{n_buildings} buildings, {len(out['walls'])} wall materials with >= 3 buildings -> {OUT}")
    return out


if __name__ == "__main__":
    res = mine()
    for w, e in list(res["walls"].items())[:12]:
        line = f"{w:26s} n={e['n']:4d}"
        for role in ("corner", "base", "frame", "roof"):
            if role in e:
                b, p = e[role]["choices"][0]
                line += f" | {role}: {b} {p:.0%} (differs {e[role]['differs']:.0%})"
        print(line)
