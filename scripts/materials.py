"""Choose materials the way a builder would: one main block per part, accents only where clearly different.

Parts:   "roof" = pitched roof surfaces (sloped, facing up); everything else (walls, flat tops, battlements,
         towers) is "body". A flat castle walkway is the same stone as its walls; a pitched roof usually isn't.
Light:   sunlit and shaded sides of the same material look different. Within each part, every surface
         direction (up / north / south / east / west) has its brightness and tint evened out to the part's
         overall level before anything is chosen.
Main:    the part's typical (median) colour -> the closest real block texture.
Accents: blocks far from the main colour (timber frames, a brick band, moss) are grouped; a group becomes an
         accent only if it is big enough and picks a different block. Lone odd blocks go back to main.
Plants:  clearly green -> leaves. Very dark -> "_dark" (windows / doorways, decided later).
"""
from collections import Counter

import numpy as np

from convert import kmeans


ODD = ("terracotta", "dripstone", "end_stone", "packed_mud", "prismarine", "nether", "concrete", "hay", "moss",
       "quartz", "mud_bricks", "polished_blackstone")


def _direction(n):
    if n[1] > 0.6:
        return "up"
    if n[1] < -0.6:
        return "down"
    return ("east" if n[0] > 0 else "west") if abs(n[0]) > abs(n[2]) else ("south" if n[2] > 0 else "north")


def assign(keys, lab, normal, shape_of, info, exclude=(), accent_de=24.0, max_accents=3, log=print, family_of=None):
    """family_of: optional callable(part, member blocks) -> list of block names the part's MAIN block may be."""
    names = [n for n in info if not any(w in n for w in exclude)]
    M = np.array([info[n][1] for n in names])
    leafy = np.array(["leaves" in n for n in names])

    # unusual blocks can be accents, but shouldn't win as a building's main block on a near-tie
    odd = np.array([6.0 if any(w in n for w in ODD) else 0.0 for n in names])

    def cost(m, main=False):
        c = np.abs(M[:, 0] - m[0]) * 0.7 + np.abs(M[:, 1] - m[1]) + np.abs(M[:, 2] - m[2])
        return c + odd if main else c

    def block(m, plant=False, main=False, allowed=None):
        c = np.where(leafy == plant, cost(m, main), np.inf)
        if allowed:
            ok = np.array([n in allowed for n in names])
            if ok.any():
                c = np.where(ok, c, np.inf)
        return names[int(np.argmin(c))]

    raw = {p: np.asarray(lab[p], float) for p in keys}
    mat = {}
    for p in keys:
        m = raw[p]
        if m[0] < 18:
            mat[p] = "_dark"
        elif m[1] < -8 and m[2] > 5 and np.hypot(m[1], m[2]) > 14:
            mat[p] = "_plant"

    # parts: pitched roof vs everything else
    part = {}
    for p in keys:
        n = normal[p]
        sloped = 0.3 < n[1] < 0.93 or shape_of[p].startswith("stairs")
        part[p] = "roof" if sloped and n[1] > 0.3 else "body"
    roof_n = sum(1 for p in keys if part[p] == "roof")
    if roof_n < 0.12 * len(keys):  # no real pitched roof (castle, flat-topped build): one part
        for p in keys:
            part[p] = "body"

    # even out light per surface direction, within each part
    norm = dict(raw)
    for pt in ("roof", "body"):
        members = [p for p in keys if part[p] == pt and p not in mat]
        if len(members) < 10:
            continue
        target = np.median(np.array([raw[p] for p in members]), 0)
        for d in ("up", "down", "north", "south", "east", "west"):
            grp = [p for p in members if _direction(normal[p]) == d]
            if len(grp) < 8:
                continue
            shift = target - np.median(np.array([raw[p] for p in grp]), 0)
            shift *= np.array([0.9, 0.6, 0.6])  # brightness fully, tint partly (sunlight is yellow)
            for p in grp:
                norm[p] = raw[p] + shift

    # camera/texture calibration: Paint's colours are darker and duller than the real blocks. Fit one
    # brightness shift + saturation gain for the whole build so its part colours line up with real blocks.
    meds = []
    for pt in ("roof", "body"):
        members = [p for p in keys if part[p] == pt and p not in mat]
        if members:
            meds.append((len(members), np.median(np.array([norm[p] for p in members]), 0)))
    nl = ~leafy

    def adj(m, sh, g):
        return np.array([m[0] + sh, m[1] * g, m[2] * g])

    allowed_of, mixed_of = {}, {}
    for pt in ("roof", "body"):
        members = [p for p in keys if part[p] == pt and p not in mat]
        if members and family_of:
            res = family_of(pt, members)
            if res:
                allowed_of[pt], mixed_of[pt] = res
    masks = []
    for pt in ("roof", "body"):
        al = allowed_of.get(pt)
        masks.append(nl & np.array([n in al for n in names]) if al else nl)

    best = min(((sh, g) for sh in range(-6, 27, 2) for g in (1.0, 1.15, 1.3, 1.45, 1.6)),
               key=lambda t: sum(n * cost(adj(m, *t), main=True)[mk].min() for (n, m), mk in zip(meds, masks))
               + abs(t[0]) * 0.3 * sum(n for n, _ in meds))
    log(f"  colour calibration: brightness {best[0]:+d}, saturation x{best[1]}")
    for p in keys:
        if p not in mat:
            norm[p] = adj(norm[p], *best)

    # main block per part, then accents
    for pt in ("roof", "body"):
        members = [p for p in keys if part[p] == pt and p not in mat]
        if not members:
            continue
        X = np.array([norm[p] for p in members])
        med = np.median(X, 0)
        allowed = allowed_of.get(pt)
        main = block(med, main=True, allowed=allowed)
        de = np.sqrt(((X - med) ** 2).sum(1))
        far = [p for p, d in zip(members, de) if d > accent_de]
        log(f"  {pt}: {len(members)} blocks, main colour L{med[0]:.0f} a{med[1]:.0f} b{med[2]:.0f} -> {main}; "
            f"{len(far)} candidates for accents")
        for p in members:
            mat[p] = main
        # two clear materials in one part (stone walls + timber frame): split before looking for accents
        if len(members) >= 40 and mixed_of.get(pt):
            g2, c2 = kmeans(X, 2, seed=1)
            sizes = [(g2 == g).mean() for g in range(2)]
            if min(sizes) >= 0.2 and np.sqrt(((c2[0] - c2[1]) ** 2).sum()) > 20:
                picks = [block(c2[g], main=True, allowed=allowed) for g in range(2)]
                if picks[0] != picks[1]:
                    log(f"    two materials: {picks[0]} {sizes[0]:.0%} + {picks[1]} {sizes[1]:.0%}")
                    for p, g in zip(members, g2):
                        mat[p] = picks[g]
                    de = np.sqrt(((X - c2[g2]) ** 2).sum(1))
                    far = [p for p, d in zip(members, de) if d > accent_de]
        if len(far) >= 6:
            F = np.array([norm[p] for p in far])
            k = min(max_accents, max(1, len(far) // 25))
            groups, cent = kmeans(F, k, seed=1)
            for g in range(len(cent)):
                sel = [p for p, gg in zip(far, groups) if gg == g]
                if len(sel) < max(6, 0.03 * len(members)):
                    continue
                acc = block(np.median(np.array([norm[p] for p in sel]), 0))
                if acc == mat[sel[0]]:
                    continue
                log(f"    accent: {len(sel)} blocks -> {acc}")
                for p in sel:
                    mat[p] = acc

    # plants -> leaves (by colour), lone plants/accents go back to their part's main block
    plants = [p for p in keys if mat[p] == "_plant"]
    if plants:
        leaf = block(np.median(np.array([raw[p] for p in plants]), 0), plant=True)
        for p in plants:
            mat[p] = leaf
    main_of = {pt: Counter(mat[p] for p in keys if part[p] == pt and not mat[p].startswith("_")).most_common(1)
               for pt in ("roof", "body")}
    for _ in range(2):
        ch = {}
        for p in keys:
            n = mat[p]
            if n.startswith("_") or not main_of[part[p]] or n == main_of[part[p]][0][0]:
                continue
            nb = [mat.get((p[0] + dx, p[1] + dy, p[2] + dz)) for dx in (-1, 0, 1) for dy in (-1, 0, 1)
                  for dz in (-1, 0, 1) if (dx, dy, dz) != (0, 0, 0)]
            if sum(1 for q in nb if q == n) < 2:
                ch[p] = main_of[part[p]][0][0]
        mat.update(ch)
    return mat, part
