"""Detailing: the general things a builder adds to any building's walls.

Works on the finished block layout (positions -> shape) and materials (positions -> block / "_glass" / "_door").
1. Windows everywhere: walls the photo couldn't see (back, far side) get windows too. A wall with none borrows
   the window layout of the wall opposite it (buildings are usually symmetric front/back); a long blank wall
   with no opposite to copy gets windows at a regular spacing at standard height.
2. A door: every building gets one ground-level doorway (2 tall) if the photo didn't show one, on the longest
   wall facing the camera.
3. Corner trim: vertical wall corners in a contrasting block that builders pair with the wall block
   (from the blockpalettes data), like log corners on plank walls or brick corners on stone.
4. Base course: the lowest wall layer in the same trim block, grounding the building.
Each rule only touches plain wall blocks (full blocks of the wall's main material): it never paints over
roof, plants, frames or other accents the photo showed.
"""
from collections import Counter

import numpy as np

DIRS = {"north": (0, 0, -1), "south": (0, 0, 1), "west": (-1, 0, 0), "east": (1, 0, 0)}
OPP = {"north": "south", "south": "north", "west": "east", "east": "west"}


def _wall_dir(p, cells):
    """Horizontal direction a wall block faces (open air on that side), or None."""
    out = [d for d, v in DIRS.items() if (p[0] + v[0], p[1] + v[1], p[2] + v[2]) not in cells]
    return out[0] if len(out) == 1 else None


def trim_block(main, contrast=(12.0, 40.0)):
    """A block builders pair with `main` that contrasts with it (palette data), or None."""
    import variety
    single, pair = variety._cooccurrence()
    m = variety.tex_lab(main)
    if m is None:
        return None
    best = None
    for (a, b), w in pair.items():
        if a != main or not variety._full(b) or any(t in b for t in variety.DECORATIVE):
            continue
        lab = variety.tex_lab(b)
        if lab is None:
            continue
        de = float(np.sqrt(((lab - m) ** 2).sum()))
        if contrast[0] <= de <= contrast[1] and (best is None or w > best[0]):
            best = (w, b)
    return best[1] if best else None


def apply(shape_of, mat, part, log=print, windows=True, door=True, corners=True, base=True):
    cells = set(shape_of)
    ys = [p[1] for p in cells]
    y0 = min(ys)
    body = [p for p in cells if part.get(p) == "body"]
    if not body:
        return mat
    main = Counter(mat[p] for p in body if not mat[p].startswith("_") and "leaves" not in mat[p]).most_common(1)
    if not main:
        return mat
    main = main[0][0]

    def plain(p):
        return p in cells and mat.get(p) == main and shape_of.get(p) == "full" and part.get(p) == "body"

    walls = {d: [] for d in DIRS}
    for p in body:
        d = _wall_dir(p, cells)
        if d and plain(p):
            walls[d].append(p)
    out = dict(mat)
    notes = []

    # 1. windows on walls without any
    glass = {d: [p for p in body if out.get(p) == "_glass" and _wall_dir(p, cells) == d] for d in DIRS}
    added = 0
    for d, ws in walls.items():
        if glass[d] or len(ws) < 12:
            continue
        src = glass[OPP[d]]
        if src:
            ax = 0 if d in ("north", "south") else 2  # coordinate along the wall
            by_line = {}
            for p in ws:
                by_line[(p[ax], p[1])] = p
            for g in src:
                q = by_line.get((g[ax], g[1]))
                if q is not None:
                    out[q] = "_glass"
                    added += 1
        else:
            ax = 0 if d in ("north", "south") else 2
            lo_, hi_ = min(p[ax] for p in ws), max(p[ax] for p in ws)
            if hi_ - lo_ < 4:
                continue
            by_line = {(p[ax], p[1]): p for p in ws}
            for c in range(lo_ + 2, hi_ - 1, 4):  # every 4 blocks, away from the corners
                for y in (y0 + 2, y0 + 3):  # standard window height
                    q = by_line.get((c, y))
                    if q is not None:
                        out[q] = "_glass"
                        added += 1
    if added:
        notes.append(f"{added} window blocks on walls the photo didn't show")

    # 2. a door if none
    if door and not any(v == "_door" for v in out.values()):
        best = max(walls.items(), key=lambda t: len(t[1]))
        d, ws = best
        if ws:
            ax = 0 if d in ("north", "south") else 2
            line = sorted({p[ax] for p in ws})
            mid = line[len(line) // 2]
            placed = 0
            for p in ws:
                if p[ax] == mid and p[1] in (y0 + 1, y0 + 2):
                    out[p] = "_door"
                    placed += 1
            if placed:
                notes.append(f"doorway on the {d} wall")

    trim = trim_block(main) if (corners or base) else None
    if trim:
        n_c = n_b = 0
        if corners:
            for p in body:
                if out.get(p) != main or shape_of.get(p) != "full":
                    continue
                open_ = [d for d, v in DIRS.items() if (p[0] + v[0], p[1] + v[1], p[2] + v[2]) not in cells]
                if len(open_) == 2 and OPP[open_[0]] != open_[1]:  # two open sides at right angles = corner
                    out[p] = trim
                    n_c += 1
        if base:
            for d, ws in walls.items():
                for p in ws:
                    if p[1] == y0 and out.get(p) == main:
                        out[p] = trim
                        n_b += 1
        notes.append(f"trim {trim}: {n_c} corner blocks, {n_b} base blocks")
    log("  detail: " + ("; ".join(notes) if notes else "nothing added"))
    return out
