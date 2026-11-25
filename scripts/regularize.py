"""Flatten a generated 3D model into clean walls and roof planes before it becomes blocks (LoD2-style).

AI-generated models are soft and bumpy; real builds are made of flat walls and straight roof slopes.
1. Align:  turn the model so its main walls run along the block grid (an angled wall becomes a jagged
           staircase of blocks). The dominant wall direction comes from an area-weighted histogram.
2. Planes: every surface is classified (wall / roof slope / flat top) and its direction snapped:
           walls to the grid axes, roof slopes to Minecraft-friendly pitches (1:2, 1:1, 2:1), flat tops level.
           Surfaces with the same snapped direction and nearly the same position form one plane.
3. Snap:   each model point moves onto its plane(s); where two or three planes meet it goes to the edge or
           corner between them. Points far from any plane (round towers, ornaments, plants) are left alone.
"""
import numpy as np
import trimesh

PITCHES = np.radians([26.565, 45.0, 63.435])  # roof slopes a builder can make: slab steps, stairs, steep


def _dominant_angle(mesh):
    n, area = mesh.face_normals, mesh.area_faces
    horiz = np.hypot(n[:, 0], n[:, 2])
    sel = horiz > 0.5
    ang = np.arctan2(n[sel, 2], n[sel, 0]) % (np.pi / 2)
    w = area[sel] * horiz[sel]
    bins = 180
    hist = np.bincount((ang / (np.pi / 2) * bins).astype(int) % bins, weights=w, minlength=bins)
    k = np.exp(-0.5 * (np.arange(-6, 7) / 2.0) ** 2)
    smooth = np.convolve(np.concatenate([hist[-6:], hist, hist[:6]]), k, "same")[6:-6]
    a = (np.argmax(smooth) + 0.5) / bins * (np.pi / 2)
    return a - np.pi / 2 if a > np.pi / 4 else a


def _snap_normal(n):
    """Snapped unit normal, or None if the surface isn't one a builder would make flat."""
    ny = n[1]
    h = np.array([n[0], 0.0, n[2]])
    hl = np.linalg.norm(h)
    if ny > 0.95:
        return np.array([0.0, 1.0, 0.0])
    if ny < -0.95:
        return np.array([0.0, -1.0, 0.0])
    if hl < 1e-6:
        return None
    hd = h / hl
    ax = np.argmax(np.abs(hd[[0, 2]]))
    if abs(hd[[0, 2]][ax]) < np.cos(np.radians(20)):
        return None  # diagonal / curved wall: leave it alone
    d = np.zeros(3)
    d[0 if ax == 0 else 2] = np.sign(hd[[0, 2]][ax])
    if abs(ny) < 0.3:
        return d  # wall
    if ny < 0:
        return None  # undersides / overhang soffits: leave
    tilt = np.arccos(np.clip(ny, -1, 1))
    j = np.argmin(np.abs(PITCHES - tilt))
    if abs(PITCHES[j] - tilt) > np.radians(12):
        return None
    t = PITCHES[j]
    return np.array([d[0] * np.sin(t), np.cos(t), d[2] * np.sin(t)])


def regularize(mesh, width, log=print, tol_blocks=0.6, min_area_blocks=3.0):
    m = mesh.copy()
    a = _dominant_angle(m)
    Rot = trimesh.transformations.rotation_matrix(a, [0, 1, 0])
    m.apply_transform(Rot)
    # after rotating by +a about Y the dominant wall direction lines up with an axis
    if abs(_dominant_angle(m)) > abs(a) + 1e-3:  # wrong sign: undo and rotate the other way
        back = trimesh.transformations.rotation_matrix(-2 * a, [0, 1, 0])
        m.apply_transform(back)
        Rot = back @ Rot
    log(f"  flatten: turned {np.degrees(a):+.1f} deg to line walls up with the block grid")

    lo, hi = m.bounds
    pitch = (hi[0] - lo[0]) / width
    tol = tol_blocks * pitch
    fn, fc, area = m.face_normals, m.triangles_center, m.area_faces

    # planes: group faces by snapped normal, then by position along that normal
    snapped = [_snap_normal(n) for n in fn]
    keys = {}
    for i, s in enumerate(snapped):
        if s is not None:
            keys.setdefault(tuple(np.round(s, 4)), []).append(i)
    planes, plane_of = [], np.full(len(fn), -1)
    for key, idx in keys.items():
        n = np.array(key)
        idx = np.array(idx)
        d = fc[idx] @ n
        order = np.argsort(d)
        start = 0
        for j in range(1, len(order) + 1):
            if j == len(order) or d[order[j]] - d[order[j - 1]] > tol * 0.5:
                grp = idx[order[start:j]]
                if area[grp].sum() >= min_area_blocks * pitch * pitch:
                    w = area[grp]
                    dd = d[order[start:j]]
                    srt = np.argsort(dd)
                    med = dd[srt][np.searchsorted(np.cumsum(w[srt]), w.sum() / 2)]
                    planes.append((n, med))
                    plane_of[grp] = len(planes) - 1
                start = j
    covered = area[plane_of >= 0].sum() / area.sum()
    kinds = {"walls": 0, "roof slopes": 0, "flat tops": 0}
    for n, _ in planes:
        kinds["flat tops" if abs(n[1]) > 0.95 else "walls" if abs(n[1]) < 0.3 else "roof slopes"] += 1
    log(f"  flatten: {len(planes)} planes ({', '.join(f'{v} {k}' for k, v in kinds.items())}) "
        f"cover {covered:.0%} of the surface")

    # snap vertices onto their plane(s) (alternating projection -> edges and corners where planes meet)
    V = m.vertices.copy()
    vf = m.vertex_faces
    moved = 0
    for vi in range(len(V)):
        fs = vf[vi][vf[vi] >= 0]
        pids = np.unique(plane_of[fs])
        pids = pids[pids >= 0]
        if len(pids) == 0:
            continue
        if len(pids) > 3:  # keep the three biggest
            pa = [area[fs][plane_of[fs] == p].sum() for p in pids]
            pids = pids[np.argsort(pa)[::-1][:3]]
        p = V[vi].copy()
        for _ in range(8):
            for pid in pids:
                n, d = planes[pid]
                p -= (p @ n - d) * n
        if np.linalg.norm(p - V[vi]) <= tol:
            V[vi] = p
            moved += 1
    m.vertices = V
    log(f"  flatten: moved {moved / len(V):.0%} of points onto planes")
    return m, Rot[:3, :3]
