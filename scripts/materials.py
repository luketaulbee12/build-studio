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
       "quartz", "mud_bricks", "polished_blackstone", "cracked", "chiseled")


def _direction(n):
    if n[1] > 0.6:
        return "up"
    if n[1] < -0.6:
        return "down"
    return ("east" if n[0] > 0 else "west") if abs(n[0]) > abs(n[2]) else ("south" if n[2] > 0 else "north")


def assign(keys, lab, normal, shape_of, info, exclude=(), accent_de=24.0, max_accents=3, log=print, family_of=None,
           faces=True, photo=None):
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

    # openings are small: a big connected dark area is deep shadow, not windows -> it takes its wall's material
    dark = {p for p in keys if mat.get(p) == "_dark"}
    seen, shadow = set(), set()
    for p0 in dark:
        if p0 in seen:
            continue
        comp, stack = [], [p0]
        seen.add(p0)
        while stack:
            p = stack.pop()
            comp.append(p)
            for q in ((p[0] + 1, p[1], p[2]), (p[0] - 1, p[1], p[2]), (p[0], p[1] + 1, p[2]),
                      (p[0], p[1] - 1, p[2]), (p[0], p[1], p[2] + 1), (p[0], p[1], p[2] - 1)):
                if q in dark and q not in seen:
                    seen.add(q)
                    stack.append(q)
        xs = np.array(comp)
        span = xs.max(0) - xs.min(0) + 1
        if len(comp) > 16 or span[1] > 5 or max(span[0], span[2]) > 6:
            shadow.update(comp)
    for p in shadow:
        del mat[p]
    if shadow:
        log(f"  {len(shadow)} dark blocks are shadow (too big for windows/doors) -> wall material")

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
        members = [p for p in keys if part[p] == pt and p not in mat and p not in shadow]
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
        members = [p for p in keys if part[p] == pt and p not in mat and p not in shadow]
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
                allowed_of[pt], mixed_of[pt] = res  # main family, second family (or None)
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

    if faces:
        def fit_gap(m, allowed):
            """How much worse the best allowed block fits than the best block of any family."""
            c = np.where(~leafy, cost(m, main=True), np.inf)
            ok = np.array([n in allowed for n in names]) if allowed else np.ones(len(names), bool)
            return float(np.where(ok, c, np.inf).min() - c.min())
        _assign_faces(keys, norm, normal, part, mat, allowed_of, mixed_of, block, log, shadow=shadow, fit_gap=fit_gap,
                      photo=photo)
    # (old way) main block per part, then accents
    for pt in (() if faces else ("roof", "body")):
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
                # the bigger group keeps the main family; the other may use the second family
                big = int(np.argmax(sizes))
                both = list(allowed or []) + list(mixed_of[pt])
                picks = [block(c2[g], main=True, allowed=allowed if g == big else both) for g in range(2)]
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


import os
_DEBUG = bool(os.environ.get("MAT_DEBUG"))


def _lighting_like(c_a, c_b):
    """True if two colours differ mostly in brightness (shadow / sun), not in hue or saturation."""
    a, b = np.asarray(c_a, float), np.asarray(c_b, float)
    if a[0] > b[0]:
        a, b = b, a  # a = the darker one
    d = b - a
    if abs(d[0]) > 2.0 * np.hypot(d[1], d[2]):
        return True  # brightness only
    ca, cb = np.hypot(a[1], a[2]), np.hypot(b[1], b[2])
    if d[0] > 8 and ca <= cb * 1.1:  # shade: darker AND no more colourful
        if ca < 8:
            return True  # shaded to grey
        hue = abs((np.degrees(np.arctan2(a[2], a[1]) - np.arctan2(b[2], b[1]) * 180 / np.pi) + 180) % 360 - 180)
        return hue < 45
    return False


def _assign_faces(keys, norm, normal, part, mat, allowed_of, mixed_of, block, log, min_face=8, shadow=(), fit_gap=None,
                  photo=None):
    """One material per wall / roof face, the way a builder thinks ("this wall is cobblestone").
    A face is a connected run of blocks of the same part facing the same way. A second material is kept on
    a face only for a large, solid region that differs in colour (not just brightness): a timber frame, a
    stone base, a brick band. Shadows, stains and scattered patches take the face's material."""
    from scipy import ndimage  # noqa: F401  (kept for clarity of dependencies)
    label = {}
    todo = [p for p in keys if p not in mat]
    key_of = {p: (part[p], _direction(normal[p])) for p in todo}
    faces = []
    for p0 in todo:
        if p0 in label:
            continue
        k = key_of[p0]
        comp, stack = [], [p0]
        label[p0] = len(faces)
        while stack:
            p = stack.pop()
            comp.append(p)
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        q = (p[0] + dx, p[1] + dy, p[2] + dz)
                        if q in key_of and q not in label and key_of[q] == k:
                            label[q] = len(faces)
                            stack.append(q)
        faces.append((k, comp))
    part_main, part_med = {}, {}
    for pt in ("roof", "body"):
        members = [p for p in todo if part[p] == pt and p not in shadow] or [p for p in todo if part[p] == pt]
        seen = [p for p in members if photo and p in photo]
        if photo and len(seen) >= 20:
            members_c = np.array([photo[p] for p in seen])
            # sunlit part of what the photo shows: the brighter half (shade only ever darkens)
            members_c = members_c[members_c[:, 0] >= np.median(members_c[:, 0])]
            part_med[pt] = np.median(members_c, 0)
            part_main[pt] = block(part_med[pt], main=True, allowed=allowed_of.get(pt))
            log(f"  {pt}: colour from the photo L{part_med[pt][0]:.0f} a{part_med[pt][1]:.0f} b{part_med[pt][2]:.0f}"
                f" -> {part_main[pt]}")
            continue
        if members:
            part_med[pt] = np.median(np.array([norm[p] for p in members]), 0)
            part_main[pt] = block(part_med[pt], main=True, allowed=allowed_of.get(pt))
            log(f"  {pt}: {len(members)} blocks, overall -> {part_main[pt]}")
    n_two, n_faces = 0, 0
    used = {}
    for (pt, d), comp in faces:
        if len(comp) < min_face:
            for p in comp:
                mat[p] = part_main[pt]
            continue
        n_faces += 1
        lit = [p for p in comp if p not in shadow] or comp
        X = np.array([norm[p] for p in comp])
        med = np.median(np.array([norm[p] for p in lit]), 0)
        seen = [p for p in comp if photo and p in photo]
        if photo and len(seen) >= max(4, 0.3 * len(comp)):
            med = np.median(np.array([photo[p] for p in seen]), 0)  # the photo shows this face: use it
            X = np.array([photo.get(p, med) for p in comp])
        for i, p in enumerate(comp):
            if p in shadow:
                X[i] = med  # unknown colour under deep shadow: assume the wall's own
        allowed = allowed_of.get(pt)
        if len(lit) < 0.5 * len(comp) or _lighting_like(med, part_med[pt])                 or np.sqrt(((med - part_med[pt]) ** 2).sum()) < 8:
            main = part_main[pt]  # same material as the building, just in shade (or practically the same)
        elif fit_gap and len(comp) >= 24 and fit_gap(med, allowed) > 25:
            main = block(med, main=True)  # colour clearly outside the labelled family (e.g. stone under "wood")
        else:
            main = block(med, main=True, allowed=allowed)
        for p in comp:
            mat[p] = main
        used[main] = used.get(main, 0) + len(comp)
        if _DEBUG:
            log(f"    face {pt}/{d}: {len(comp)} blocks, median L{med[0]:.0f} a{med[1]:.0f} b{med[2]:.0f} -> {main}")
        if len(comp) < 24 or X.std(0).max() < 1e-6:
            continue
        from convert import kmeans as _km
        g2, c2 = _km(X, 2, seed=1)
        sizes = [(g2 == g).sum() for g in range(2)]
        big = int(np.argmax(sizes))
        sm = 1 - big
        if sizes[sm] < max(6, 0.15 * len(comp)):
            continue
        if _DEBUG:
            log(f"      split {sizes} L{c2[big][0]:.0f}/{c2[sm][0]:.0f} b{c2[big][2]:.0f}/{c2[sm][2]:.0f}")
        if np.sqrt(((c2[sm] - c2[big]) ** 2).sum()) < 20 or _lighting_like(c2[sm], c2[big]):
            continue
        sel = [p for p, g in zip(comp, g2) if g == sm]
        # solid regions only: drop scattered bits (connected pieces smaller than 4 blocks)
        sset, keep = set(sel), []
        seen = set()
        for p0 in sel:
            if p0 in seen:
                continue
            piece, stack = [], [p0]
            seen.add(p0)
            while stack:
                p = stack.pop()
                piece.append(p)
                for q in ((p[0] + 1, p[1], p[2]), (p[0] - 1, p[1], p[2]), (p[0], p[1] + 1, p[2]),
                          (p[0], p[1] - 1, p[2]), (p[0], p[1], p[2] + 1), (p[0], p[1], p[2] - 1)):
                    if q in sset and q not in seen:
                        seen.add(q)
                        stack.append(q)
            if len(piece) >= 4:
                keep += piece
        if len(keep) < max(6, 0.1 * len(comp)):
            continue
        both = list(allowed or []) + list(mixed_of.get(pt) or [])
        second = block(c2[sm], main=True, allowed=both if mixed_of.get(pt) else None)
        if second == main:
            continue
        for p in keep:
            mat[p] = second
        n_two += 1
        used[second] = used.get(second, 0) + len(keep)
    log(f"  faces: {n_faces} wall/roof faces, {n_two} with a second material; "
        + ", ".join(f"{k} {v}" for k, v in sorted(used.items(), key=lambda t: -t[1])[:5]))
