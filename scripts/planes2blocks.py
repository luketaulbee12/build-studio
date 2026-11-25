"""Rebuild the block layout from clean geometry (LoD2-style): straight walls, roofs as exact planes.

Works on the block grid after the model has been lined up with it.
Walls:  fill one-block notches and gaps, remove one-block bumps, so wall faces run straight.
Roofs:  the top surface (highest block of every column) is split into planes, the way city models are made:
        flat, or sloping along one axis at a pitch a builder can make (1:2 slab steps, 1:1 stairs, 2:1 steep).
        Each plane is fitted greedily to the columns it explains best; then those columns are rebuilt exactly:
        full blocks up to the plane, and the top block becomes a stair (1:1) or a slab (half steps),
        facing uphill. Columns no plane explains (towers, chimneys, ornaments) keep the model's shape.
"""
from collections import deque

import numpy as np
from scipy import ndimage

SLOPES = (0.5, 1.0, 2.0)


def straighten(occ):
    """occ: bool [x, y, z]. Returns a copy with notches/gaps filled and single bumps removed."""
    o = occ.copy()
    k8 = np.ones((3, 3), int); k8[1, 1] = 0
    for y in range(o.shape[1]):
        layer = o[:, y, :]
        n = ndimage.convolve(layer.astype(int), k8, mode="constant")
        layer |= (~layer) & (n >= 6)  # notch in a straight wall
        o[:, y, :] = layer
    # vertical gaps of one block in a wall
    gap = (~o[:, 1:-1, :]) & o[:, :-2, :] & o[:, 2:, :]
    o[:, 1:-1, :] |= gap
    # one-block bumps: attached to the wall on one side only, nothing above or below
    k4 = np.zeros((3, 3), int); k4[0, 1] = k4[1, 0] = k4[1, 2] = k4[2, 1] = 1
    above = np.zeros_like(o); above[:, :-1, :] = o[:, 1:, :]
    below = np.zeros_like(o); below[:, 1:, :] = o[:, :-1, :]
    n4 = np.stack([ndimage.convolve(o[:, y, :].astype(int), k4, mode="constant") for y in range(o.shape[1])], 1)
    bump = o & (n4 == 1) & ~above & ~below
    o &= ~bump
    return o, int(((o != occ)).sum())


def fit_roofs(occ, min_cells=8, max_change=1.5, log=print):
    """Returns (new occupancy, {cell: shape}) with the top surface rebuilt from fitted planes."""
    X, Y, Z = occ.shape
    has = occ.any(1)
    top = np.where(has, Y - 1 - np.argmax(occ[:, ::-1, :], 1), -1)  # y of highest block per column
    H = (top + 1).astype(float)  # surface height (top face of the highest block)
    free = has.copy()
    xs, zs = np.meshgrid(np.arange(X), np.arange(Z), indexing="ij")
    models = [(0.0, 0, 1)] + [(s, ax, sg) for s in SLOPES for ax in (0, 2) for sg in (1, -1)]
    planes = []
    while True:
        best = None
        for s, ax, sg in models:
            coord = (xs if ax == 0 else zs) * sg
            c = H - s * coord
            vals = np.round(c[free] * 2) / 2
            if len(vals) == 0:
                break
            u, cnt = np.unique(vals, return_counts=True)
            for cc in u[np.argsort(-cnt)[:3]]:
                inl = free & (np.abs(H - (cc + s * coord)) <= 0.5)
                lab, nl = ndimage.label(inl)
                if nl == 0:
                    continue
                sizes = np.bincount(lab.ravel())[1:]
                j = int(np.argmax(sizes)) + 1
                n = sizes[j - 1]
                if best is None or n > best[0]:
                    best = (n, s, ax, sg, cc, lab == j)
        if best is None or best[0] < min_cells:
            break
        n, s, ax, sg, cc, region = best
        planes.append((s, ax, sg, cc, region))
        free &= ~region
    new = occ.copy()
    shape = {}
    changed = 0
    for s, ax, sg, cc, region in planes:
        coord = (xs if ax == 0 else zs) * sg
        h = cc + s * coord
        uphill = {(0, 1): "east", (0, -1): "west", (2, 1): "south", (2, -1): "north"}[(ax, sg)]
        for x, z in zip(*np.where(region)):
            hv = h[x, z]
            if abs(hv - H[x, z]) > max_change:
                continue
            col = new[x, :, z]
            bottom = int(np.argmax(col)) if col.any() else 0
            if s == 0:
                full_to = int(round(hv))
                part = None
            elif s == 1.0:
                full_to = int(np.floor(hv + 1e-6)) - 1  # top block becomes a stair
                part = f"stairs_{uphill}_bottom"
            elif s == 0.5:
                full_to = int(np.floor(hv + 1e-6))
                part = "slab_bottom" if hv - full_to >= 0.5 - 1e-6 else None
            else:  # steep 2:1 - full blocks, rounded
                full_to = int(round(hv))
                part = None
            # rebuild only the column's top: below the lower of old/new surface the model is kept as is
            old_top = top[x, z]
            for y in range(max(bottom, min(old_top, full_to)), Y):
                want = y < full_to or (part is not None and y == full_to)
                if part is not None and y == full_to:
                    shape[(x, y, z)] = part
                if new[x, y, z] != want:
                    changed += 1
                    new[x, y, z] = want
    kinds = {}
    for s, *_ in planes:
        k = "flat" if s == 0 else f"{'1:2' if s == 0.5 else '1:1' if s == 1 else '2:1'} slope"
        kinds[k] = kinds.get(k, 0) + 1
    covered = sum(r.sum() for *_, r in planes) / max(1, has.sum())
    log(f"  roofs: {len(planes)} planes ({', '.join(f'{v} {k}' for k, v in kinds.items())}) explain {covered:.0%} "
        f"of the top surface; {changed} blocks changed")
    return new, shape
