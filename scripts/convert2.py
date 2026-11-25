"""Reconstructed 3D model + picture -> Minecraft build, done the way a builder would.

usage: python scripts/convert2.py <mesh.glb> <picture> <name> --width 16 [--at X Z] [--out dir]

1. Camera:    simplified model, every triangle drawn; best silhouette match, refined to 1 degree.
2. Shape:     block grid from the model (thin stacked parts kept), hollow shell.
3. Parts:     split the picture into a few materials (superpixels grouped by colour + texture), so a wall
              is one material instead of a patchwork.
4. Blocks:    each material -> one Minecraft block (real texture colour + texture busyness). Every block
              takes the material it shows most of; hidden sides take the main material of their part.
5. Cleanup:   lone odd blocks join their neighbours; roof slopes become stairs when the roof block has them.
6. Score:     render from the picture's camera, compare, save picture | build side by side.
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pymeshlab
import trimesh
from PIL import Image
from scipy.spatial import cKDTree
from skimage.segmentation import slic

sys.path.insert(0, str(Path(__file__).resolve().parent))
from convert import BLOCKS, LEAF_TINT, TEX, STUDIO, SUB, fit_to_mask, kmeans, rot, silhouette, to_lab  # noqa: E402
from matte import matte  # noqa: E402
import render  # noqa: E402


# ---------------------------------------------------------------------------------- textures

def texture_info():
    """name -> (mean RGB, mean Lab, L std) from the real block textures."""
    info = {}
    for name, (tex, _, _) in BLOCKS.items():
        f = TEX / f"{tex}.png"
        if not f.exists():
            continue
        px = np.asarray(Image.open(f).convert("RGBA")).astype(np.float64)[:16]
        rgb = px[..., :3][px[..., 3] > 0]
        if "leaves" in name:
            rgb = rgb * LEAF_TINT / 0.55
        rgb = np.clip(rgb, 0, 255)
        lab = to_lab(rgb)
        info[name] = (rgb.mean(0), lab.mean(0), lab[:, 0].std())
    return info


# ---------------------------------------------------------------------------------- stages

def fit_camera(mesh, mask):
    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(mesh.vertices, mesh.faces))
    ms.meshing_decimation_quadric_edge_collapse(targetfacenum=30000, preservetopology=False)
    m = ms.current_mesh()
    V, F = m.vertex_matrix(), m.face_matrix()
    ys, xs = np.where(mask)
    box = (xs.min(), xs.max(), ys.min(), ys.max())

    def iou(elev, yaw):
        R = rot(elev, yaw)
        sil = silhouette(V, F, R, fit_to_mask(V, R, box), mask.shape)
        return (sil & mask).sum() / max(1, (sil | mask).sum())

    best = max(((iou(e, y), e, y) for e in range(0, 41, 4) for y in range(-30, 31, 5)))
    _, e0, y0 = best
    best = max(((iou(e, y), e, y) for e in range(max(0, e0 - 3), e0 + 4) for y in range(y0 - 4, y0 + 5)))
    s, e, y = best
    R = rot(e, y)
    return s, e, y, R, fit_to_mask(V, R, box)


def voxelize(mesh, width):
    lo, hi = mesh.bounds
    pitch = (hi[0] - lo[0]) / width
    sub = pitch / SUB
    vox = mesh.voxelized(sub).fill()
    occ = vox.matrix
    origin = vox.indices_to_points(np.array([[0, 0, 0]]))[0] - sub / 2
    nx, ny, nz = (np.array(occ.shape) + SUB - 1) // SUB
    padded = np.zeros((nx * SUB, ny * SUB, nz * SUB), bool)
    padded[:occ.shape[0], :occ.shape[1], :occ.shape[2]] = occ
    b = padded.reshape(nx, SUB, ny, SUB, nz, SUB)
    frac = b.mean((1, 3, 5))
    solid = frac >= 0.45
    thin = (frac >= 0.2) & ~solid
    up = np.zeros_like(thin); up[:, :-1] = frac[:, 1:] >= 0.2
    dn = np.zeros_like(thin); dn[:, 1:] = frac[:, :-1] >= 0.2
    solid |= thin & up & dn
    # thin horizontal parts (eaves, porch roofs, ledges): keep as slabs when they spread sideways
    some = frac >= 0.25
    nb = sum(np.roll(some, sh, ax) for ax in (0, 2) for sh in (1, -1))
    half_top = b[:, :, :, SUB // 2:, :, :].mean((1, 3, 5)) > b[:, :, :, :SUB // 2 + 1, :, :].mean((1, 3, 5))
    slab = some & ~solid & (nb >= 3)
    solid |= slab
    pad = np.pad(solid, 1)
    inner = pad[2:, 1:-1, 1:-1] & pad[:-2, 1:-1, 1:-1] & pad[1:-1, 2:, 1:-1] & pad[1:-1, :-2, 1:-1] \
        & pad[1:-1, 1:-1, 2:] & pad[1:-1, 1:-1, :-2]
    return solid, solid & ~inner, origin, pitch, slab, half_top


def picture_materials(rgb, lab, mask, k):
    """Superpixels grouped into k materials. Returns per-pixel material label (-1 outside)."""
    seg = slic(rgb.astype(np.uint8), n_segments=500, compactness=12, mask=mask, start_label=1)
    ids = np.unique(seg[seg > 0])
    feat = []
    for i in ids:
        px = lab[seg == i]
        feat.append(np.concatenate([px.mean(0), [px[:, 0].std() * 0.7]]))
    feat = np.array(feat)
    groups, _ = kmeans(feat, k, seed=1)
    label = np.full(mask.shape, -1)
    for i, g in zip(ids, groups):
        label[seg == i] = g
    return label


def choose_blocks(lab, label, k, info, allow):
    """One block per picture material. Brightness shift searched globally (picture lighting)."""
    stats = []
    for g in range(k):
        px = lab[label == g]
        stats.append((len(px), px.mean(0) if len(px) else np.zeros(3), px[:, 0].std() if len(px) else 0))
    names = [n for n in info if n in allow]
    M = np.array([info[n][1] for n in names])
    S = np.array([info[n][2] for n in names])

    tot = sum(n for n, _, _ in stats)
    Lmean = sum(n * m[0] for n, m, _ in stats) / max(1, tot)
    leafy = np.array(["leaves" in n for n in names])

    def cost(shift, m, s, con=1.0):
        # picture lighting: brightness shift + contrast (renders shade harder than block textures do)
        L = Lmean + (m[0] - Lmean) * con + shift
        return np.abs(M[:, 1] - m[1]) + np.abs(M[:, 2] - m[2]) + np.abs(M[:, 0] - L) * 0.6 \
            + np.abs(S - s * con) * 0.4

    building = [(n, m, s) for n, m, s in stats if n and m[0] >= 20 and not (m[1] < -7 and m[2] > 5)]
    shift, con = min(((sh, c) for sh in range(-10, 41, 2) for c in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0)),
                     key=lambda t: sum(n * cost(t[0], m, s, t[1])[~leafy].min() for n, m, s in building))
    print(f"lighting fit: brightness {shift:+d}, contrast x{con}")
    out = {}
    for g, (n, m, s) in enumerate(stats):
        if not n:
            continue
        c = cost(shift, m, s, con)
        if m[0] < 20:  # deep shadow / openings: not a material
            out[g] = ("_dark", c, names)
            continue
        plant = m[1] < -7 and m[2] > 5
        c = np.where(leafy == plant, c, np.inf)  # green foliage -> leaves; everything else never leaves
        out[g] = (names[int(np.argmin(c))], c, names)
    return shift, out, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mesh")
    ap.add_argument("picture")
    ap.add_argument("name")
    ap.add_argument("--width", type=int, default=16)
    ap.add_argument("--materials", type=int, default=8)
    ap.add_argument("--at", type=int, nargs=2, default=None, metavar=("X", "Z"))
    ap.add_argument("--world", default=r"C:\Users\Luke\Desktop\Projects\freecam-ghost\neoforge\run\saves\New World")
    ap.add_argument("--out", default="out/dbg")
    a = ap.parse_args()
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    mesh = trimesh.load(a.mesh, force="mesh")
    pic = np.asarray(matte(a.picture, pad_ratio=1.0)).astype(np.float64)
    rgb, mask = pic[..., :3], pic[..., 3] > 127

    iou, elev, yaw, R, fit = fit_camera(mesh, mask)
    print(f"camera: elevation {elev}°, yaw {yaw}° (silhouette match {iou:.2f})")

    solid, shell, origin, pitch, slab, half_top = voxelize(mesh, a.width)
    print(f"thin overhangs kept as slabs: {int(slab.sum())}")
    idx = [tuple(int(v) for v in p) for p in np.argwhere(shell)]
    cells = {p: None for p in idx}
    block_px = pitch * fit[0]
    print(f"grid {solid.shape}, shell {len(idx)} blocks, ~{block_px:.0f} px per block in the picture")

    # which blocks the picture shows, and which pixels
    own, keys, fdir = render.owners(cells, R, fit, mask.shape, origin, pitch)
    own[~mask] = -1
    fdir[~mask] = -1

    # roles from the model's surface direction
    centres = origin + (np.array(keys) + 0.5) * pitch
    _, near = cKDTree(mesh.triangles_center).query(centres)
    normals = mesh.face_normals[near]
    H = solid.shape[1]
    role = {}
    for (x, y, z), n in zip(keys, normals):
        above_empty = y + 1 >= H or not solid[x, y + 1, z]
        role[(x, y, z)] = "roof" if above_empty and n[1] > 0.3 and y >= H * 0.35 else "wall"
    normal_of = dict(zip(keys, normals))

    # picture -> materials -> blocks
    # lighting: side walls in shade get the front wall's light (same material, different light)
    lab = to_lab(rgb)
    wall_px = np.zeros(mask.shape, bool)
    for i, p in enumerate(keys):
        if role[p] == "wall":
            pass
    wall_ids = np.array([role[p] == "wall" for p in keys])
    wall_px = (own >= 0) & wall_ids[np.maximum(own, 0)]
    front = wall_px & (fdir == 4)
    for side in (0, 1):
        sel = wall_px & (fdir == side)
        if sel.sum() > 500 and front.sum() > 500:
            d = np.median(lab[front], 0) - np.median(lab[sel], 0)
            d[0] *= 0.7
            print(f"lighting: side {side} corrected by L{d[0]:+.1f} a{d[1]:+.1f} b{d[2]:+.1f}")
            lab[(fdir == side) & mask] += d
    k = a.materials
    label = picture_materials(rgb, lab, mask, k)
    info = texture_info()
    allow = set(info)
    shift, pick, stats = choose_blocks(lab, label, k, info, allow)
    print(f"brightness shift +{shift}")
    for g, (name, _, _) in sorted(pick.items()):
        n, m, s = stats[g]
        print(f"  material {g}: {n:6d} px  L{m[0]:5.1f} a{m[1]:6.1f} b{m[2]:6.1f} texture {s:4.1f} -> {name}")

    # each visible block: the material it shows most of
    mat = {}
    dark_of = {}
    for i in np.unique(own[own >= 0]):
        labs = label[own == i]
        labs = labs[labs >= 0]
        if not len(labs):
            continue
        cnt = Counter(labs.tolist())
        dark = sum(v for g, v in cnt.items() if pick[g][0] == "_dark")
        if dark >= 0.6 * len(labs):
            dark_of[keys[i]] = True  # window / doorway / deep shadow
            continue
        mat[keys[i]] = next(g for g, _ in cnt.most_common() if pick[g][0] != "_dark")
    print(f"visible blocks {len(mat)} / {len(keys)}")

    # cleanup: lone blocks join their neighbours (same part, 26-neighbourhood)
    for _ in range(2):
        changes = {}
        for p, g in mat.items():
            nb = [mat[q] for q in ((p[0] + dx, p[1] + dy, p[2] + dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1)
                                   for dz in (-1, 0, 1) if (dx, dy, dz) != (0, 0, 0))
                  if q in mat and role[q] == role[p]]
            if len(nb) >= 3:
                top, cnt = Counter(nb).most_common(1)[0]
                if g not in nb and cnt >= 0.5 * len(nb):
                    changes[p] = top
        mat.update(changes)
        print(f"cleanup pass: {len(changes)} lone blocks merged")

    # block names; hidden blocks take their part's main block
    name_of = {p: pick[g][0] for p, g in mat.items()}
    main = {}
    for r in ("roof", "wall"):
        c = Counter(n for p, n in name_of.items() if role[p] == r and "leaves" not in n)
        main[r] = c.most_common(1)[0][0] if c else "cobblestone"
    for p in keys:
        name_of.setdefault(p, main[role[p]])
    print("main blocks:", main, f"| dark (window/door/shadow) blocks: {len(dark_of)}")

    # roof: needs stairs for slopes - if the roof block has none, the closest-looking block that does
    roof_names = Counter(n for p, n in name_of.items() if role[p] == "roof")
    stair_sub = {}
    for n in roof_names:
        if BLOCKS[n][1] is None and "leaves" not in n:
            g = next(g for g, v in pick.items() if v[0] == n)
            _, c, names = pick[g]
            order = np.argsort(c)
            best = c[order[0]]
            alt = next((names[j] for j in order if BLOCKS[names[j]][1] and "leaves" not in names[j]), None)
            if alt and c[names.index(alt)] <= best * 1.3 + 4:
                stair_sub[n] = alt
    if stair_sub:
        print("roof stairs from:", stair_sub)

    states = {}
    for p in keys:
        n = name_of[p]
        nrm = normal_of[p]
        s = "minecraft:" + n
        if "leaves" in n:
            s += "[persistent=true]"
        elif role[p] == "roof" and np.hypot(nrm[0], nrm[2]) > 0.25 and nrm[1] < 0.95:
            base = stair_sub.get(n, n)
            stairs = BLOCKS[base][1]
            if stairs:
                facing = (("east" if nrm[0] < 0 else "west") if abs(nrm[0]) > abs(nrm[2])
                          else ("south" if nrm[2] < 0 else "north"))
                s = f"minecraft:{stairs}[facing={facing},half=bottom]"
                name_of[p] = base
        elif "log" in n:
            s += "[axis=y]"
        if slab[p] and "leaves" not in n and "stairs" not in s:
            base = stair_sub.get(n, n)
            if BLOCKS[base][2]:
                s = f"minecraft:{BLOCKS[base][2]}[type={'top' if half_top[p] else 'bottom'}]"
                name_of[p] = base
        states[p] = s
    # openings: dark wall areas -> windows (glass panes), or doorways if they reach the ground
    ymin = min(p[1] for p in keys)
    seen = set()
    for p0 in [p for p in dark_of if role[p] == "wall"]:
        if p0 in seen:
            continue
        comp, q = [], [p0]
        seen.add(p0)
        while q:
            p = q.pop()
            comp.append(p)
            for d in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
                nq = (p[0] + d[0], p[1] + d[1], p[2] + d[2])
                if nq in dark_of and nq not in seen and role[nq] == "wall":
                    seen.add(nq)
                    q.append(nq)
        door = min(p[1] for p in comp) <= ymin + 1
        for p in comp:
            states[p] = "minecraft:air" if door else "minecraft:glass_pane"
            name_of[p] = "_door" if door else "_glass"
    print("blocks used:", dict(Counter(name_of.values()).most_common()))

    # score + side by side
    colour = {**{n: v[0] for n, v in info.items()}, "_glass": (40, 45, 50), "_door": (15, 15, 15)}
    for n in colour:
        if "leaves" in n:
            colour[n] = np.asarray(colour[n]) * 0.75
    ren, rmask = render.render(name_of, lambda n: colour[n], R, fit, mask.shape, origin, pitch)
    s_iou, s_err = render.score(rgb, mask, ren, rmask, block_px)
    print(f"SCORE: shape match {s_iou:.3f}, colour error {s_err:.1f} (lower is better)")
    safe = a.name.replace(" ", "_")
    render.side_by_side(rgb.astype(np.uint8), ren, out_dir / f"{safe}_compare.png")
    lab_img = np.zeros((*mask.shape, 3), np.uint8)
    pal = np.array([info[pick[g][0]][0] if g in pick and pick[g][0] in info else (20, 20, 30) for g in range(k)], np.uint8)
    lab_img[label >= 0] = pal[label[label >= 0]]
    Image.fromarray(lab_img).save(out_dir / f"{safe}_materials.png")
    (out_dir / f"{safe}_score.json").write_text(json.dumps({"iou": s_iou, "colour_error": s_err}))

    if a.at:
        export(states, a)


def export(states, a):
    sys.path.insert(0, str(STUDIO))
    import studio
    from region import RegionReader, region_dir
    ground_y = getattr(a, "ground", None)
    rr = None if ground_y is not None else RegionReader(region_dir(a.world, "minecraft_overworld"))
    not_ground = ("air", "leaves", "_log", "grass", "fern", "flower", "bush", "vine", "poppy", "dandelion",
                  "tulip", "daisy", "bluet", "orchid", "allium", "cornflower", "lily", "water", "snow")

    def ground_at(x, z):
        y = rr.surface(x, z)
        while y > -64 and any(t in rr.block(x, y, z) for t in not_ground):
            y -= 1
        return y

    pts = np.array(list(states))
    mn = pts.min(0)
    wx, wz = int(pts[:, 0].max() - mn[0] + 1), int(pts[:, 2].max() - mn[2] + 1)
    gx, gz = a.at
    x0, z0 = gx - wx // 2, gz - wz // 2
    if ground_y is not None:  # no terrain to read (superflat preview / any server): sit on this height
        ground = int(ground_y)
    else:
        ground = min(ground_at(x0 + dx, z0 + dz) for dx in range(0, wx, 2) for dz in range(0, wz, 2))
    out, taken = [], set()
    for p, s in states.items():
        x, y, z = x0 + p[0] - int(mn[0]), ground + 1 + p[1] - int(mn[1]), z0 + p[2] - int(mn[2])
        out.append([x, y, z, s])
        taken.add((x, y, z))
    top = ground + 1 + int(pts[:, 1].max() - mn[1]) + 8
    for x in (range(x0 - 3, x0 + wx + 3) if rr else ()):
        for z in range(z0 - 3, z0 + wz + 3):
            for y in range(ground + 1, top):
                if (x, y, z) not in taken and any(t in rr.block(x, y, z) for t in ("leaves", "_log", "grass", "fern", "bush")):
                    out.append([x, y, z, "minecraft:air"])
    cfg = studio.load_config()
    game_dir = getattr(a, "game_dir", None) or cfg["gameDirs"][0]
    world_id = getattr(a, "world_id", None) or "sp_" + studio.sanitize(Path(a.world).name)
    res = studio.export_blueprint({"gameDir": game_dir, "worldId": world_id,
                                   "dimension": "minecraft_overworld", "name": a.name, "blocks": out}, cfg)
    print("blueprint:", res)


if __name__ == "__main__":
    main()
