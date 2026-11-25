"""Painted 3D model -> Minecraft build at true scale, using partial blocks for detail.

usage: python scripts/build_from_model.py <painted.glb> <picture> <name> --width 16 [--palette 8] [--at X Z]

--width is the whole model's width in blocks (the picture's build, including bushes/props).
1. Shape:     sample the model 4x4x4 inside every block; pick the best-fitting Minecraft shape
              (full, slab, stairs, fence/wall post, trapdoor plate, floor, thin wall).
2. Colour:    the painted texture inside each block, colour balance matched to the picture.
3. Materials: a builder-sized palette (k groups), each -> the closest real block texture.
4. Blocks:    shape x material -> block state (stairs of that material, its slab, fence/wall, trapdoor...).
              Dark wall areas become windows (glass panes) or doorways. Fences/walls/panes get connections.
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from convert import BLOCKS, kmeans, to_lab  # noqa: E402
from convert2 import export, fit_camera, texture_info  # noqa: E402
from matte import matte  # noqa: E402
from voxelize_tex import surface_colours  # noqa: E402
import render  # noqa: E402
import shapes  # noqa: E402
import materials  # noqa: E402

N = shapes.N
NB6 = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))


def cell_occupancy(mesh, width):
    lo, hi = mesh.bounds
    pitch = (hi[0] - lo[0]) / width
    sub = pitch / N
    vox = mesh.voxelized(sub).fill()
    occ = vox.matrix.copy()
    # models with an open underside (hidden by bushes etc.) don't fill: close small gaps, then fill each
    # horizontal layer's enclosed interior so walls come out solid
    from scipy import ndimage
    occ = ndimage.binary_closing(occ, structure=np.ones((3, 3, 3)), iterations=1) | occ
    for y in range(occ.shape[1]):
        occ[:, y, :] = ndimage.binary_fill_holes(occ[:, y, :])
    origin = vox.indices_to_points(np.array([[0, 0, 0]]))[0] - sub / 2
    dims = (np.array(occ.shape) + N - 1) // N
    pad = np.zeros(dims * N, bool)
    pad[:occ.shape[0], :occ.shape[1], :occ.shape[2]] = occ
    cells = pad.reshape(dims[0], N, dims[1], N, dims[2], N).transpose(0, 2, 4, 1, 3, 5)
    return cells, origin, pitch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mesh")
    ap.add_argument("picture")
    ap.add_argument("name")
    ap.add_argument("--width", type=int, default=16)
    ap.add_argument("--palette", type=int, default=8)
    ap.add_argument("--variety", type=float, default=0.35, help="share of main blocks swapped for related blocks (0 = off)")
    ap.add_argument("--exclude", default="", help="comma-separated words: leave out blocks containing them")
    ap.add_argument("--at", type=int, nargs=2, default=None, metavar=("X", "Z"))
    ap.add_argument("--world", default=r"C:\Users\Luke\Desktop\Projects\freecam-ghost\neoforge\run\saves\New World")
    ap.add_argument("--out", default="out/dbg")
    a = ap.parse_args()
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    mesh = trimesh.load(a.mesh, force="mesh")
    pic = np.asarray(matte(a.picture, pad_ratio=1.0)).astype(np.float64)
    rgb_pic, mask = pic[..., :3], pic[..., 3] > 127

    # --- shapes
    cells, origin, pitch = cell_occupancy(mesh, a.width)
    frac = cells.mean((3, 4, 5))
    cand = np.argwhere(frac >= 0.08)
    names, score = shapes.match(cells[tuple(cand.T)])
    shape_of = {}
    for p, n, s in zip(map(tuple, cand), names, score):
        f = frac[p]
        if s < 0.2:  # nothing fits well: solid enough -> full block, otherwise leave it out
            if f < 0.5:
                continue
            n = "full"
        shape_of[tuple(int(v) for v in p)] = n
    full = {p for p, n in shape_of.items() if n == "full"}
    interior = {p for p in full if all((p[0] + d[0], p[1] + d[1], p[2] + d[2]) in full for d in NB6)}
    for p in interior:
        del shape_of[p]
    keys = list(shape_of)
    keyset = set(keys)
    print(f"grid {frac.shape}: {len(keys)} blocks ({len(interior)} hidden inside removed); shapes:",
          dict(Counter(shape_of.values()).most_common()))

    # --- colours from the painted texture (colour balance matched to the picture)
    from skimage.color import lab2rgb
    pts, cols, nrm = surface_colours(mesh)
    pl, sl = to_lab(rgb_pic[mask]), to_lab(cols)
    for ch in (1, 2):
        sl[:, ch] = (sl[:, ch] - sl[:, ch].mean()) / max(1e-6, sl[:, ch].std()) * pl[:, ch].std() + pl[:, ch].mean()
    lab_pts = sl
    cell_idx = np.floor((pts - origin) / pitch).astype(int)
    per = {}
    for c, l, n in zip(map(tuple, cell_idx), lab_pts, nrm):
        per.setdefault(c, []).append((l, n))
    tree = cKDTree(np.array([k for k in per]))
    have = list(per)
    lab, normal = {}, {}
    for p in keys:
        src = per.get(p)
        if src is None:  # no surface inside (inner side of a partial block): nearest surface cell
            src = per[have[tree.query(p)[1]]]
        lab[p] = np.median(np.array([v[0] for v in src]), 0)
        nn = np.array([v[1] for v in src]).mean(0)
        normal[p] = nn / max(1e-9, np.linalg.norm(nn))

    # --- materials (one main block per part, accents only where clearly different)
    info = texture_info()
    skip = [w for w in a.exclude.split(",") if w]
    # what each part is made of, from the picture (needs the camera to know which pixels are which part)
    iou, elev, yaw, R, fit = fit_camera(mesh, mask)
    own, okeys, _ = render.owners({p: 1 for p in keys}, R, fit, mask.shape, origin, pitch)
    own[~mask] = -1
    oindex = {p: i for i, p in enumerate(okeys)}
    import material_family as mf

    def family_of(pt, members):
        ids = np.array([oindex[p] for p in members])
        pm = np.isin(own, ids)
        fams = mf.ROOF if pt == "roof" else mf.BODY
        best, score = mf.classify(rgb_pic.astype(np.uint8), pm, fams)
        if best is None:
            return None
        print(f"  {pt} looks like: " + ", ".join(f"{k} {v:.0%}" for k, v in sorted(score.items(), key=lambda t: -t[1])[:3]))
        return mf.allowed_blocks(fams, score)

    mat, part = materials.assign(keys, lab, normal, shape_of, info, exclude=skip, family_of=family_of)
    base_mat = dict(mat)
    if a.variety > 0:
        import variety
        mat = variety.apply(mat, keys, amount=a.variety)

    # --- block states
    main = Counter(n for n in mat.values() if not n.startswith("_") and "leaves" not in n).most_common(1)[0][0]
    ymin = min(p[1] for p in keys)
    states, family = {}, {}
    for p in keys:
        n, shp = mat[p], shape_of[p]
        if n == "_dark":
            vertical = abs(normal[p][1]) < 0.5
            if vertical and shp in ("full", "plate_north", "plate_south", "plate_east", "plate_west", "pane_x", "pane_z"):
                if p[1] <= ymin + 2:
                    states[p], mat[p] = "minecraft:air", "_door"
                else:
                    states[p], family[p], mat[p] = "minecraft:glass_pane", "pane", "_glass"
                continue
            n = mat[p] = main
        s, fam = shapes.block_for(shp, n, BLOCKS)
        if shp != "full" and n != base_mat.get(p, n) and (s is None or "stairs" in shp and "stairs" not in s):
            n = mat[p] = base_mat[p]  # variant has no stair/slab form: keep the main block for shaped pieces
            s, fam = shapes.block_for(shp, n, BLOCKS)
        if s is None:  # this material has no such shape: nearest sensible fallback
            if cells[p].mean() >= 0.4 or shp.startswith(("plate", "pane")):  # thin walls are still walls
                s, fam = shapes.block_for("full", n, BLOCKS)
                shape_of[p] = "full"
            else:
                continue
        states[p] = s
        if fam:
            family[p] = fam
    solid_full = {p for p, s in states.items() if shape_of.get(p) == "full" and "leaves" not in s and s != "minecraft:air"}
    states = shapes.connect(states, family, solid_full)
    kinds = Counter(s.split("[")[0].replace("minecraft:", "") for s in states.values())
    print("blocks used:", dict(kinds.most_common(18)))

    # --- score against the picture (simple cube render)
    info_col = {**{n: v[0] for n, v in info.items()}, "_glass": (40, 45, 50), "_door": (15, 15, 15)}
    shown = {p: mat[p] for p in states}
    ren, rmask = render.render(shown, lambda n: info_col.get(n, (128, 128, 128)), R, fit, mask.shape, origin, pitch)
    s_iou, s_err = render.score(rgb_pic, mask, ren, rmask, pitch * fit[0])
    print(f"SCORE: shape match {s_iou:.3f}, colour error {s_err:.1f}")
    safe = a.name.replace(" ", "_")
    render.side_by_side(rgb_pic.astype(np.uint8), ren, out_dir / f"{safe}_compare.png")
    (out_dir / f"{safe}_score.json").write_text(json.dumps({"iou": s_iou, "colour_error": s_err}))
    if a.at:
        export(states, a)


if __name__ == "__main__":
    main()
