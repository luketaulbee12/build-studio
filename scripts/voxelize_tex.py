"""Textured 3D model -> Minecraft blocks, the way ObjToSchematic-style converters do it, plus Minecraft parts.

usage: python scripts/voxelize_tex.py <textured.glb> <picture> <name> --width 32 [--palette 8] [--at X Z]

1. Shape:    block grid from the model (thin stacked parts kept, thin overhangs as slabs), hollow shell.
2. Colour:   sample the model's own painted texture densely over its surface; each block gets the colours
             of the surface inside it (all sides are painted, lighting already removed by Paint).
3. Blocks:   group block colours into a builder-sized palette (--palette, 0 = match each block on its own),
             each group -> the Minecraft block whose real texture matches best.
4. Parts:    roof slopes -> stairs, thin overhangs -> slabs, dark wall areas -> windows / doorways.
5. Score:    render from the picture's camera and compare with the picture (side by side image).
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image
from trimesh.visual.color import uv_to_color

sys.path.insert(0, str(Path(__file__).resolve().parent))
from convert import BLOCKS, kmeans, to_lab  # noqa: E402
from convert2 import export, fit_camera, texture_info, voxelize  # noqa: E402
from matte import matte  # noqa: E402
import render  # noqa: E402


def surface_colours(mesh, n=400_000):
    pts, fi = trimesh.sample.sample_surface_even(mesh, n, seed=0)
    tri = mesh.triangles[fi]
    bary = trimesh.triangles.points_to_barycentric(tri, pts)
    uv = (mesh.visual.uv[mesh.faces[fi]] * bary[:, :, None]).sum(1)
    tex = mesh.visual.material.baseColorTexture if hasattr(mesh.visual.material, "baseColorTexture") \
        else mesh.visual.material.image
    rgb = uv_to_color(uv, tex)[:, :3].astype(np.float64)
    return pts, rgb, mesh.face_normals[fi]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mesh")
    ap.add_argument("picture")
    ap.add_argument("name")
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--palette", type=int, default=8)
    ap.add_argument("--at", type=int, nargs=2, default=None, metavar=("X", "Z"))
    ap.add_argument("--world", default=r"C:\Users\Luke\Desktop\Projects\freecam-ghost\neoforge\run\saves\New World")
    ap.add_argument("--out", default="out/dbg")
    a = ap.parse_args()
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    mesh = trimesh.load(a.mesh, force="mesh")
    pic = np.asarray(matte(a.picture, pad_ratio=1.0)).astype(np.float64)
    rgb_pic, mask = pic[..., :3], pic[..., 3] > 127

    solid, shell, origin, pitch, slab, half_top = voxelize(mesh, a.width)
    keys = [tuple(int(v) for v in p) for p in np.argwhere(shell)]
    keyset = set(keys)
    print(f"grid {solid.shape}, shell {len(keys)} blocks")

    # colours from the painted surface
    pts, cols, nrm = surface_colours(mesh)
    # Paint's lighting-free texture has a colour cast of its own: match its colour (a, b) spread to the
    # picture's, keep Paint's brightness
    from skimage.color import lab2rgb
    pl = to_lab(rgb_pic[mask])
    sl = to_lab(cols)
    for ch in (1, 2):
        sl[:, ch] = (sl[:, ch] - sl[:, ch].mean()) / max(1e-6, sl[:, ch].std()) * pl[:, ch].std() + pl[:, ch].mean()
    print(f"colour balance: picture a{pl[:, 1].mean():+.1f} b{pl[:, 2].mean():+.1f}, "
          f"paint was a{to_lab(cols)[:, 1].mean():+.1f} b{to_lab(cols)[:, 2].mean():+.1f}")
    cols = np.clip(lab2rgb(sl) * 255, 0, 255)
    cell = np.floor((pts - origin) / pitch).astype(int)
    per = {}
    for c, col, n in zip(map(tuple, cell), cols, nrm):
        if c not in keyset:  # surface point just outside the kept block: give it to a kept neighbour
            c = next((q for q in ((c[0] + dx, c[1] + dy, c[2] + dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1)
                                  for dz in (-1, 0, 1)) if q in keyset), None)
            if c is None:
                continue
        per.setdefault(c, []).append((col, n))
    colour, normal = {}, {}
    for p in keys:
        if p in per:
            cs = np.array([v[0] for v in per[p]])
            colour[p] = np.median(cs, 0)
            nn = np.array([v[1] for v in per[p]]).mean(0)
            normal[p] = nn / max(1e-9, np.linalg.norm(nn))
    # blocks with no surface inside (rare): nearest coloured neighbour
    missing = [p for p in keys if p not in colour]
    if missing:
        have = np.array([p for p in colour])
        from scipy.spatial import cKDTree
        _, j = cKDTree(have).query(np.array(missing))
        for p, jj in zip(missing, j):
            q = tuple(have[jj])
            colour[p], normal[p] = colour[q], normal[q]
    print(f"coloured from the painted texture: {len(per)} blocks direct, {len(missing)} borrowed")

    # roles
    H = solid.shape[1]
    role = {}
    for (x, y, z) in keys:
        n = normal[(x, y, z)]
        above_empty = y + 1 >= H or not solid[x, y + 1, z]
        role[(x, y, z)] = "roof" if above_empty and n[1] > 0.3 and y >= H * 0.35 else "wall"

    # palette
    info = texture_info()
    names = list(info)
    M = np.array([info[n][1] for n in names])
    leafy = np.array(["leaves" in n for n in names])
    lab = {p: to_lab(colour[p]) for p in keys}

    def best_block(m):
        plant = m[1] < -7 and m[2] > 5
        c = np.abs(M[:, 0] - m[0]) * 0.7 + np.abs(M[:, 1] - m[1]) + np.abs(M[:, 2] - m[2])
        c = np.where(leafy == plant, c, np.inf)
        return names[int(np.argmin(c))], c

    name_of, dark = {}, set()
    X = np.array([lab[p] for p in keys])
    if a.palette:
        groups, cent = kmeans(X, a.palette, seed=1)
        pick = {}
        for g in range(len(cent)):
            if (groups == g).any():
                m = X[groups == g].mean(0)
                pick[g] = "_dark" if m[0] < 18 else best_block(m)[0]
                print(f"  group {g}: {int((groups == g).sum()):5d} blocks  L{m[0]:5.1f} a{m[1]:6.1f} b{m[2]:6.1f} -> {pick[g]}")
        for p, g in zip(keys, groups):
            name_of[p] = pick[g]
    else:
        for p, m in zip(keys, X):
            name_of[p] = "_dark" if m[0] < 18 else best_block(m)[0]

    # lone odd blocks join their neighbours (same part)
    for _ in range(2):
        ch = {}
        for p, n in name_of.items():
            nb = [name_of[q] for q in ((p[0] + dx, p[1] + dy, p[2] + dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1)
                                       for dz in (-1, 0, 1) if (dx, dy, dz) != (0, 0, 0)) if q in name_of and role[q] == role[p]]
            if len(nb) >= 3 and n not in nb:
                top, cnt = Counter(nb).most_common(1)[0]
                if cnt >= 0.5 * len(nb):
                    ch[p] = top
        name_of.update(ch)

    # dark: windows/doorways on walls, otherwise the part's main block
    main = {r: (Counter(n for p, n in name_of.items() if role[p] == r and not n.startswith("_") and "leaves" not in n)
                .most_common(1) or [("cobblestone", 0)])[0][0] for r in ("roof", "wall")}
    ymin = min(p[1] for p in keys)
    states = {}
    for p in keys:
        if name_of[p] == "_dark":
            if role[p] == "wall":
                below = (p[0], p[1] - 1, p[2])
                door = p[1] <= ymin + 1 or (name_of.get(below) == "_dark" and below[1] <= ymin + 1)
                states[p] = "minecraft:air" if door else "minecraft:glass_pane"
                name_of[p] = "_door" if door else "_glass"
                continue
            name_of[p] = main[role[p]]

    # Minecraft parts
    for p in keys:
        if p in states:
            continue
        n, nrm_ = name_of[p], normal[p]
        s = "minecraft:" + n
        if "leaves" in n:
            s += "[persistent=true]"
        elif role[p] == "roof" and np.hypot(nrm_[0], nrm_[2]) > 0.25 and nrm_[1] < 0.95 and BLOCKS[n][1]:
            facing = (("east" if nrm_[0] < 0 else "west") if abs(nrm_[0]) > abs(nrm_[2])
                      else ("south" if nrm_[2] < 0 else "north"))
            s = f"minecraft:{BLOCKS[n][1]}[facing={facing},half=bottom]"
        elif slab[p] and BLOCKS[n][2]:
            s = f"minecraft:{BLOCKS[n][2]}[type={'top' if half_top[p] else 'bottom'}]"
        elif "log" in n:
            s += "[axis=y]"
        states[p] = s
    print("blocks used:", dict(Counter(name_of.values()).most_common()))

    # score against the picture
    iou, elev, yaw, R, fit = fit_camera(mesh, mask)
    print(f"camera: elevation {elev}, yaw {yaw} (silhouette match {iou:.2f})")
    colour_of = {**{n: v[0] for n, v in info.items()}, "_glass": (40, 45, 50), "_door": (15, 15, 15)}
    for n in colour_of:
        if "leaves" in n:
            colour_of[n] = np.asarray(colour_of[n]) * 0.75
    ren, rmask = render.render(name_of, lambda n: colour_of[n], R, fit, mask.shape, origin, pitch)
    s_iou, s_err = render.score(rgb_pic, mask, ren, rmask, pitch * fit[0])
    print(f"SCORE: shape match {s_iou:.3f}, colour error {s_err:.1f} (lower is better)")
    safe = a.name.replace(" ", "_")
    render.side_by_side(rgb_pic.astype(np.uint8), ren, out_dir / f"{safe}_compare.png")
    (out_dir / f"{safe}_score.json").write_text(json.dumps({"iou": s_iou, "colour_error": s_err}))

    if a.at:
        export(states, a)


if __name__ == "__main__":
    main()
