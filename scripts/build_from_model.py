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
    orig = occ.copy()
    occ = ndimage.binary_closing(occ, structure=np.ones((3, 3, 3)), iterations=1) | occ
    for y in range(occ.shape[1]):
        occ[:, y, :] = ndimage.binary_fill_holes(occ[:, y, :])
    # only enclosed space under a roof is "inside": open courtyards (sky above) stay empty
    roofed = np.flip(np.cumsum(np.flip(orig, 1), 1), 1) > orig  # some model surface higher up this column
    occ = orig | (occ & roofed)
    origin = vox.indices_to_points(np.array([[0, 0, 0]]))[0] - sub / 2
    dims = (np.array(occ.shape) + N - 1) // N
    pad = np.zeros(dims * N, bool)
    pad[:occ.shape[0], :occ.shape[1], :occ.shape[2]] = occ
    cells = pad.reshape(dims[0], N, dims[1], N, dims[2], N).transpose(0, 2, 4, 1, 3, 5)
    return cells, origin, pitch


_SUB = {}


def _stair_sub(name):
    """Closest-coloured block (by real texture) that has stairs and slabs, for materials without them."""
    if name in _SUB:
        return _SUB[name]
    import variety
    m = variety.tex_lab(name)
    best = None
    if m is not None:
        for n, (_, st, sl) in BLOCKS.items():
            if not st or not sl or "leaves" in n:
                continue
            lab_ = variety.tex_lab(n)
            if lab_ is None:
                continue
            d = float(np.sqrt(((lab_ - m) ** 2).sum()))
            if best is None or d < best[0]:
                best = (d, n)
    _SUB[name] = best[1] if best else None
    return _SUB[name]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mesh")
    ap.add_argument("picture")
    ap.add_argument("name")
    ap.add_argument("--width", type=int, default=16)
    ap.add_argument("--palette", type=int, default=8)
    ap.add_argument("--ground", type=int, default=None, help="place on this height instead of reading the world's terrain")
    ap.add_argument("--game-dir", default=None, help="game folder to save the blueprint into (default: first in studio config)")
    ap.add_argument("--world-id", default=None, help="blueprint world folder id, e.g. sp_New_World")
    ap.add_argument("--tops", type=int, default=0, help="1 = even out ragged tops no roof plane explains")
    ap.add_argument("--bridge", type=int, default=1, help="1 = fill one-block gaps in thin sheets")
    ap.add_argument("--windens", type=int, default=0, help="1 = generated windows match the photo's window density")
    ap.add_argument("--learned", type=int, default=0, help="1 = corner/base trim from habits learned from real builds")
    ap.add_argument("--doors", type=int, default=1, help="1 = put real doors in doorways")
    ap.add_argument("--detail", type=int, default=1, help="1 = add windows on unseen walls, a door, corner/base trim")
    ap.add_argument("--snap", type=int, default=-1, help="1 = face colours snap to the picture's own main colours, 0 = never, -1 = automatically for Minecraft screenshots")
    ap.add_argument("--photo-faces", type=int, default=0, help="1 = faces the photo shows take the photo's colour")
    ap.add_argument("--faces", type=int, default=1, help="1 = one material per wall/roof face (0 = older per-part rule)")
    ap.add_argument("--photo-colour", type=int, default=0, help="1 = shift colours toward the photo (experimental)")
    ap.add_argument("--planes", type=int, default=1, help="1 = rebuild straight walls / exact roof planes")
    ap.add_argument("--flat", type=int, default=1, help="1 = flatten walls / roof planes first (0 = raw model)")
    ap.add_argument("--iso", default=None, help="also save an overview render of the build to this file")
    ap.add_argument("--variety", type=float, default=0.35, help="share of main blocks swapped for related blocks (0 = off)")
    ap.add_argument("--exclude", default="", help="comma-separated words: leave out blocks containing them")
    ap.add_argument("--at", type=int, nargs=2, default=None, metavar=("X", "Z"))
    ap.add_argument("--world", default=None, help="world folder to read the ground from (with --at); without it the build sits on --ground")
    ap.add_argument("--out", default="out/dbg")
    a = ap.parse_args()
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    mesh = trimesh.load(a.mesh, force="mesh")
    mesh0, Rot = mesh, np.eye(3)
    if a.flat:
        import regularize
        mesh, Rot = regularize.regularize(mesh, a.width)
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
    if a.bridge:
        # thin sheets (tin roofs, boards, thin walls) only partly fill the blocks they cross, so some blocks
        # drop out and the sheet falls apart. A weakly filled block with kept blocks on two opposite sides
        # (straight or diagonal) is the missing piece of a sheet: keep it.
        kept = set(shape_of)
        bridged = 0
        dirs_ = ((1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, -1, 0), (0, 1, 1), (0, 1, -1), (1, 0, 1), (1, 0, -1))
        weak = [tuple(int(v) for v in q) for q in np.argwhere(frac >= 0.04)]
        for _ in range(2):
            new = []
            for q in weak:
                if q in kept:
                    continue
                for d in dirs_:
                    if (q[0] + d[0], q[1] + d[1], q[2] + d[2]) in kept and (q[0] - d[0], q[1] - d[1], q[2] - d[2]) in kept:
                        new.append(q)
                        break
            for q in new:
                shape_of[q] = "full"
                kept.add(q)
            bridged += len(new)
        print(f"  thin parts: {bridged} gap blocks bridged")
    if a.planes:
        import planes2blocks
        occ = np.zeros(frac.shape, bool)
        for p in shape_of:
            occ[p] = True
        occ2, fixed = planes2blocks.straighten(occ)
        print(f"  walls: {fixed} blocks changed to straighten wall faces")
        occ3, roof_shape = planes2blocks.fit_roofs(occ2)
        if a.tops:
            occ3 = planes2blocks.even_tops(occ3, roof_shape)
        new_shape = {}
        for p in map(tuple, np.argwhere(occ3)):
            p = tuple(int(v) for v in p)
            if p in roof_shape:
                new_shape[p] = roof_shape[p]
            elif p in shape_of and not (occ3[p[0], p[1] + 1:, p[2]].any() is False and shape_of[p].startswith("stairs")):
                new_shape[p] = shape_of[p]
            else:
                new_shape[p] = "full"
        # a surface block the planes rebuilt as full keeps full (no stray stair from the soft model)
        for p in list(new_shape):
            if new_shape[p].startswith(("stairs", "slab")) and p not in roof_shape:
                above = (p[0], p[1] + 1, p[2])
                if above in new_shape:
                    new_shape[p] = "full"
        shape_of = new_shape
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
    for ch in (1, 2):  # Paint's colours are duller than the photo: match their spread (keeps greens green)
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
    iou, elev, yaw, R, fit = fit_camera(mesh0, mask)
    R = R @ Rot.T  # same camera, expressed for the turned (flattened) model
    own, okeys, _ = render.owners({p: 1 for p in keys}, R, fit, mask.shape, origin, pitch)
    own[~mask] = -1
    oindex = {p: i for i, p in enumerate(okeys)}

    # the photo is the colour reference: for blocks the photo shows, compare the model's paint with the photo
    # and fit one colour correction per channel for the whole build (robust line fit); hidden sides follow it
    pic_lab = to_lab(rgb_pic)
    flat_own = own.ravel()
    order = np.argsort(flat_own)
    bounds = np.searchsorted(flat_own[order], np.arange(len(okeys) + 1))
    pl_all = pic_lab.reshape(-1, 3)
    pairs = []
    photo_col = {}
    for i, q in enumerate(okeys):
        idx = order[bounds[i]:bounds[i + 1]]
        if len(idx) >= 15:
            photo_col[q] = np.median(pl_all[idx], 0)
            pairs.append((lab[q], photo_col[q]))
    if a.photo_colour and len(pairs) >= 30:
        A = np.array([x for x, _ in pairs]); B = np.array([y for _, y in pairs])
        # a plain shift per channel (median of photo - paint): robust to the noisy block-by-block pairing
        cfit = [(1.0, float(np.median(B[:, ch] - A[:, ch]))) for ch in range(3)]
        for q in keys:
            lab[q] = np.array([cfit[ch][0] * lab[q][ch] + cfit[ch][1] for ch in range(3)])
        print("  colour from the photo: " + ", ".join(f"{n} x{k:.2f}{c0:+.1f}" for n, (k, c0) in zip("Lab", cfit))
              + f" (from {len(pairs)} visible blocks)")
    import material_family as mf

    def family_of(pt, members):
        ids = np.array([oindex[p] for p in members])
        pm = np.isin(own, ids)
        fams = mf.ROOF if pt == "roof" else mf.BODY
        if pt == "roof":
            whole = rgb_pic.astype(np.uint8).copy()
            whole[~mask] = 255
            from PIL import Image as _Im
            best, score = mf.classify_roof(rgb_pic.astype(np.uint8), pm, _Im.fromarray(whole))
        else:
            best, score = mf.classify(rgb_pic.astype(np.uint8), pm, fams)
        if best is None:
            return None
        print(f"  {pt} looks like: " + ", ".join(f"{k} {v:.0%}" for k, v in sorted(score.items(), key=lambda t: -t[1])[:3]))
        return mf.allowed_blocks(fams, score)

    palette = None
    if a.snap < 0:  # auto: a game screenshot's colours are the real block colours; a photo's include lighting
        from PIL import Image as _Im2
        mc = mf.is_minecraft(_Im2.open(a.picture).convert("RGB"))
        a.snap = int(mc > 0.6)
        print(f"  picture looks like a Minecraft screenshot: {mc:.2f} -> {'use' if a.snap else 'do not use'} its colours")
    if a.snap:
        # the photo's own main colours (by area, deep shade left out); Paint picks among them per face
        px = to_lab(rgb_pic[mask])
        px = px[px[:, 0] >= np.percentile(px[:, 0], 20)]
        sub = px[np.random.default_rng(0).choice(len(px), min(20000, len(px)), replace=False)]
        g, cent = kmeans(sub, 6, seed=2)
        share = np.bincount(g, minlength=len(cent)) / len(g)
        palette = [c for c, sh in zip(cent, share) if sh >= 0.04]
        print("  photo palette: " + ", ".join(f"L{c[0]:.0f} a{c[1]:.0f} b{c[2]:.0f}" for c in palette))
    mat, part = materials.assign(keys, lab, normal, shape_of, info, exclude=skip, family_of=family_of, palette=palette,
                                 faces=bool(a.faces), photo=photo_col if a.photo_faces else None)
    # dark wall areas -> windows / doorways (decided here so detailing can see them)
    ymin0 = min(q[1] for q in keys)
    body_main = Counter(v for v in mat.values() if not v.startswith("_") and "leaves" not in v).most_common(1)
    body_main = body_main[0][0] if body_main else "cobblestone"
    for q in keys:
        if mat[q] == "_dark":
            vertical = abs(normal[q][1]) < 0.5
            if vertical and shape_of[q] in ("full", "plate_north", "plate_south", "plate_east", "plate_west",
                                            "pane_x", "pane_z"):
                mat[q] = "_door" if q[1] <= ymin0 + 2 else "_glass"
            else:
                mat[q] = body_main
    if a.detail:
        import detail
        mat = detail.apply(shape_of, mat, part, match_density=bool(a.windens), learned=bool(a.learned))
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
        if n == "_door":
            states[p] = "minecraft:air"
            continue
        if n == "_glass":
            states[p], family[p] = "minecraft:glass_pane", "pane"
            continue
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
        if s is None and shp.startswith(("stairs", "slab")):
            n2 = _stair_sub(n)  # slopes need stairs: closest-coloured block that has a stair / slab form
            if n2:
                s, fam = shapes.block_for(shp, n2, BLOCKS)
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
    # doorways get real doors: bottom two blocks of each doorway column, facing out, in the building's wood
    door_cells = sorted(p for p in states if mat.get(p) == "_door")
    if door_cells and a.doors:
        woods = Counter(shapes.wood_of(v) for v in mat.values() if shapes.wood_of(v))
        wood = woods.most_common(1)[0][0] if woods else "spruce"
        cols = {}
        for p in door_cells:
            cols.setdefault((p[0], p[2]), []).append(p[1])
        cellset = set(states)
        n_doors = 0
        for (x, z), ys_ in cols.items():
            ys_ = sorted(ys_)
            if len(ys_) < 2 or ys_[1] != ys_[0] + 1:
                continue
            out_ = [d for d, (dx, dz) in (("north", (0, -1)), ("south", (0, 1)), ("west", (-1, 0)), ("east", (1, 0)))
                    if (x + dx, ys_[0], z + dz) not in cellset or mat.get((x + dx, ys_[0], z + dz)) == "_door"
                    and False]
            facing = out_[0] if out_ else "south"
            states[(x, ys_[0], z)] = f"minecraft:{wood}_door[facing={facing},half=lower,hinge=left,open=false,powered=false]"
            states[(x, ys_[0] + 1, z)] = f"minecraft:{wood}_door[facing={facing},half=upper,hinge=left,open=false,powered=false]"
            n_doors += 1
        print(f"  doors: {n_doors} {wood} door(s)")
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
    if a.iso:
        from convert import fit_to_mask, rot
        Ri = rot(28, 35)
        corners = origin + np.array([[x + dx, y + dy, z + dz] for (x, y, z) in states for dx in (0, 1)
                                     for dy in (0, 1) for dz in (0, 1)][::4]) * pitch
        fi = fit_to_mask(corners, Ri, (20, 460, 20, 460))
        import variety
        from skimage.color import lab2rgb

        def col(n):
            if n in info_col:
                return info_col[n]
            lab_ = variety.tex_lab(n)
            return (128, 128, 128) if lab_ is None else np.clip(lab2rgb(lab_[None])[0] * 255, 0, 255)
        cells_s = {p: (mat[p], shape_of.get(p, "full")) for p in states if not mat[p].startswith("_door")}
        img, _ = render.render_shapes(cells_s, col, Ri, fi, (480, 480), origin, pitch,
                               bg=(170, 200, 235))
        from PIL import Image as _I
        _I.fromarray(img).save(a.iso)
    if a.at:
        export(states, a)


if __name__ == "__main__":
    main()
