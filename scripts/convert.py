"""Reconstructed 3D model + the original picture -> a Minecraft build (blueprint).

usage: python scripts/convert.py <mesh.glb> <picture> <name> --width 16 --at X Z

1. Shape:     sample the model 3x finer than a block; a block is solid if the model fills most of it.
2. Camera:    find the angle the picture was taken from by matching silhouettes.
3. Colours:   for every block the camera can see, average the picture pixels it covers; hidden blocks
              borrow from the nearest visible block.
4. Materials: group the colours into a few materials (like a builder would), pick one Minecraft block
              per group by comparing against the real block textures.
5. Parts:     sloped tops become stairs, half-filled tops become slabs.
"""
import argparse
import sys
from collections import deque
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from matte import matte  # noqa: E402

STUDIO = Path(__file__).resolve().parent.parent / "studio"  # Build Studio lives next to scripts/
TEX = STUDIO / "data" / "mc" / "textures" / "block"
SUB = 3  # sub-samples per block edge

# candidate block -> (texture file, stairs variant or None, slab variant or None)
BLOCKS = {
    "stone": ("stone", "stone_stairs", "stone_slab"),
    "cobblestone": ("cobblestone", "cobblestone_stairs", "cobblestone_slab"),
    "mossy_cobblestone": ("mossy_cobblestone", "mossy_cobblestone_stairs", "mossy_cobblestone_slab"),
    "stone_bricks": ("stone_bricks", "stone_brick_stairs", "stone_brick_slab"),
    "mossy_stone_bricks": ("mossy_stone_bricks", "mossy_stone_brick_stairs", "mossy_stone_brick_slab"),
    "cracked_stone_bricks": ("cracked_stone_bricks", None, None),
    "andesite": ("andesite", "andesite_stairs", "andesite_slab"),
    "polished_andesite": ("polished_andesite", "polished_andesite_stairs", "polished_andesite_slab"),
    "diorite": ("diorite", "diorite_stairs", "diorite_slab"),
    "granite": ("granite", "granite_stairs", "granite_slab"),
    "tuff": ("tuff", "tuff_stairs", "tuff_slab"),
    "tuff_bricks": ("tuff_bricks", "tuff_brick_stairs", "tuff_brick_slab"),
    "calcite": ("calcite", None, None),
    "deepslate_bricks": ("deepslate_bricks", "deepslate_brick_stairs", "deepslate_brick_slab"),
    "deepslate_tiles": ("deepslate_tiles", "deepslate_tile_stairs", "deepslate_tile_slab"),
    "cobbled_deepslate": ("cobbled_deepslate", "cobbled_deepslate_stairs", "cobbled_deepslate_slab"),
    "polished_deepslate": ("polished_deepslate", "polished_deepslate_stairs", "polished_deepslate_slab"),
    "bricks": ("bricks", "brick_stairs", "brick_slab"),
    "mud_bricks": ("mud_bricks", "mud_brick_stairs", "mud_brick_slab"),
    "sandstone": ("sandstone", "sandstone_stairs", "sandstone_slab"),
    "smooth_stone": ("smooth_stone", None, "smooth_stone_slab"),
    "oak_planks": ("oak_planks", "oak_stairs", "oak_slab"),
    "spruce_planks": ("spruce_planks", "spruce_stairs", "spruce_slab"),
    "birch_planks": ("birch_planks", "birch_stairs", "birch_slab"),
    "dark_oak_planks": ("dark_oak_planks", "dark_oak_stairs", "dark_oak_slab"),
    "jungle_planks": ("jungle_planks", "jungle_stairs", "jungle_slab"),
    "acacia_planks": ("acacia_planks", "acacia_stairs", "acacia_slab"),
    "stripped_oak_log": ("stripped_oak_log", None, None),
    "stripped_spruce_log": ("stripped_spruce_log", None, None),
    "stripped_dark_oak_log": ("stripped_dark_oak_log", None, None),
    "spruce_log": ("spruce_log", None, None),
    "oak_log": ("oak_log", None, None),
    "white_terracotta": ("white_terracotta", None, None),
    "light_gray_terracotta": ("light_gray_terracotta", None, None),
    "terracotta": ("terracotta", None, None),
    "azalea_leaves": ("azalea_leaves", None, None),
    "oak_leaves": ("oak_leaves", None, None),
    "moss_block": ("moss_block", None, "moss_carpet"),
    "glass": ("glass", None, None),
    "smooth_sandstone": ("sandstone_top", "smooth_sandstone_stairs", "smooth_sandstone_slab"),
    "cut_sandstone": ("cut_sandstone", None, "cut_sandstone_slab"),
    "red_sandstone": ("red_sandstone", "red_sandstone_stairs", "red_sandstone_slab"),
    "cut_red_sandstone": ("cut_red_sandstone", None, "cut_red_sandstone_slab"),
    "end_stone_bricks": ("end_stone_bricks", "end_stone_brick_stairs", "end_stone_brick_slab"),
    "dripstone_block": ("dripstone_block", None, None),
    "packed_mud": ("packed_mud", None, None),
    "polished_granite": ("polished_granite", "polished_granite_stairs", "polished_granite_slab"),
    "polished_diorite": ("polished_diorite", "polished_diorite_stairs", "polished_diorite_slab"),
    "quartz_block": ("quartz_block_side", "quartz_stairs", "quartz_slab"),
    "blackstone": ("blackstone", "blackstone_stairs", "blackstone_slab"),
    "polished_blackstone_bricks": ("polished_blackstone_bricks", "polished_blackstone_brick_stairs", "polished_blackstone_brick_slab"),
    "nether_bricks": ("nether_bricks", "nether_brick_stairs", "nether_brick_slab"),
    "prismarine_bricks": ("prismarine_bricks", "prismarine_brick_stairs", "prismarine_brick_slab"),
    "white_concrete": ("white_concrete", None, None),
    "hay_block": ("hay_block_side", None, None),
    "birch_log": ("birch_log", None, None),
    "dark_oak_log": ("dark_oak_log", None, None),
    "stripped_birch_log": ("stripped_birch_log", None, None),
}
LEAF_TINT = np.array([0x77, 0xAB, 0x2F]) / 255.0


# ---------------------------------------------------------------------------------- colour helpers

def to_lab(c):
    c = np.asarray(c, np.float64) / 255.0
    c = np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)
    m = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
    xyz = c @ m.T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def texture_stats():
    """(names, mean Lab, L std) for every candidate block's real texture."""
    names, means, stds = [], [], []
    for name, (tex, _, _) in BLOCKS.items():
        f = TEX / f"{tex}.png"
        if not f.exists():
            continue
        px = np.asarray(Image.open(f).convert("RGBA")).astype(np.float64)[:16]
        a = px[..., 3] > 0
        rgb = px[..., :3][a]
        if "leaves" in name:
            rgb = rgb * LEAF_TINT / 0.55
        lab = to_lab(np.clip(rgb, 0, 255))
        names.append(name)
        means.append(lab.mean(0))
        stds.append(lab[:, 0].std())
    return names, np.array(means), np.array(stds)


def kmeans(x, k, iters=30, seed=0):
    rng = np.random.default_rng(seed)
    c = [x[rng.integers(len(x))]]
    for _ in range(1, k):
        d = np.min(((x[:, None] - np.array(c)[None]) ** 2).sum(-1), 1)
        c.append(x[rng.choice(len(x), p=d / d.sum())])
    c = np.array(c)
    for _ in range(iters):
        lab = np.argmin(((x[:, None] - c[None]) ** 2).sum(-1), 1)
        for j in range(k):
            if (lab == j).any():
                c[j] = x[lab == j].mean(0)
    return lab, c


# ---------------------------------------------------------------------------------- geometry

def rot(elev, yaw):
    """Rotation taking model coords into camera coords (camera looks down -Z, Y up)."""
    ry = trimesh.transformations.rotation_matrix(np.radians(yaw), [0, 1, 0])[:3, :3]
    rx = trimesh.transformations.rotation_matrix(np.radians(elev), [1, 0, 0])[:3, :3]
    return rx @ ry


def project(points, R, fit):
    """Model points -> picture pixel coords using rotation R and the bbox fit (sx, ox, sy, oy)."""
    p = points @ R.T
    sx, ox, sy, oy = fit
    return np.stack([p[:, 0] * sx + ox, -p[:, 1] * sy + oy], 1), p[:, 2]


def fit_to_mask(verts, R, mask_box):
    p = verts @ R.T
    x0, x1, y0, y1 = mask_box
    sx = (x1 - x0) / max(1e-9, p[:, 0].max() - p[:, 0].min())
    sy = (y1 - y0) / max(1e-9, p[:, 1].max() - p[:, 1].min())
    s = (sx + sy) / 2
    return (s, x0 - p[:, 0].min() * s, s, y0 + p[:, 1].max() * s)


def silhouette(verts, faces, R, fit, shape):
    uv, _ = project(verts, R, fit)
    img = Image.new("1", (shape[1], shape[0]), 0)
    d = ImageDraw.Draw(img)
    step = max(1, len(faces) // 40000)
    for f in faces[::step]:
        d.polygon([tuple(uv[i]) for i in f], fill=1)
    return np.asarray(img, bool)


def find_camera(mesh, mask):
    ys, xs = np.where(mask)
    box = (xs.min(), xs.max(), ys.min(), ys.max())
    best = None
    for front in (0,):  # Hunyuan always turns the object's pictured side toward +Z
        base = trimesh.transformations.rotation_matrix(front * np.pi / 2, [0, 1, 0])[:3, :3]
        for elev in range(0, 41, 5):
            for yaw in range(-30, 31, 10):
                R = rot(elev, yaw) @ base
                fit = fit_to_mask(mesh.vertices, R, box)
                sil = silhouette(mesh.vertices, mesh.faces, R, fit, mask.shape)
                iou = (sil & mask).sum() / max(1, (sil | mask).sum())
                if best is None or iou > best[0]:
                    best = (iou, front, elev, yaw, R, fit)
    return best


# ---------------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mesh")
    ap.add_argument("picture")
    ap.add_argument("name")
    ap.add_argument("--width", type=int, default=16, help="blocks across the front")
    ap.add_argument("--materials", type=int, default=7)
    ap.add_argument("--at", type=int, nargs=2, required=True, metavar=("X", "Z"))
    ap.add_argument("--world", default=None, help="world folder to read the ground from (with --at); without it the build sits on --ground")
    ap.add_argument("--debug", default=None, help="folder for debug images")
    a = ap.parse_args()
    dbg = Path(a.debug) if a.debug else None
    if dbg:
        dbg.mkdir(parents=True, exist_ok=True)

    mesh = trimesh.load(a.mesh, force="mesh")
    pic = matte(a.picture, pad_ratio=1.0)
    rgba = np.asarray(pic).astype(np.float64)
    mask = rgba[..., 3] > 127

    # --- camera
    iou, front, elev, yaw, R, fit = find_camera(mesh, mask)
    print(f"camera: front side {front}, elevation {elev}°, yaw {yaw}°  (silhouette IoU {iou:.2f})")
    base = trimesh.transformations.rotation_matrix(front * np.pi / 2, [0, 1, 0])
    mesh.apply_transform(base)  # model now faces +Z toward the camera
    R = rot(elev, yaw)

    # --- shape: sample SUB^3 points per block, keep blocks mostly inside the model
    lo, hi = mesh.bounds
    pitch = (hi[0] - lo[0]) / a.width
    sub = pitch / SUB
    vox = mesh.voxelized(sub).fill()
    occ = vox.matrix
    origin = vox.indices_to_points(np.array([[0, 0, 0]]))[0] - sub / 2
    nx, ny, nz = (np.array(occ.shape) + SUB - 1) // SUB
    padded = np.zeros((nx * SUB, ny * SUB, nz * SUB), bool)
    padded[:occ.shape[0], :occ.shape[1], :occ.shape[2]] = occ
    blocks = padded.reshape(nx, SUB, ny, SUB, nz, SUB)
    frac = blocks.mean((1, 3, 5))
    lower = blocks[:, :, :, :SUB // 2 + 1, :, :].mean((1, 3, 5))  # how full the bottom part is
    upper = blocks[:, :, :, SUB // 2:, :, :].mean((1, 3, 5))
    solid = frac >= 0.45
    # thin vertical parts (chimneys, posts) never fill a block - keep them if they stack
    thin = (frac >= 0.2) & ~solid
    up = np.zeros_like(thin); up[:, :-1] = (frac[:, 1:] >= 0.2)
    dn = np.zeros_like(thin); dn[:, 1:] = (frac[:, :-1] >= 0.2)
    solid |= thin & up & dn
    print(f"grid {solid.shape} ({pitch:.4f} model units per block), solid blocks {int(solid.sum())}")

    # keep a shell (anything with an empty neighbour) - houses are hollow
    pad = np.pad(solid, 1)
    inner = pad[2:, 1:-1, 1:-1] & pad[:-2, 1:-1, 1:-1] & pad[1:-1, 2:, 1:-1] & pad[1:-1, :-2, 1:-1] \
        & pad[1:-1, 1:-1, 2:] & pad[1:-1, 1:-1, :-2]
    shell = solid & ~inner
    idx = np.argwhere(shell)
    centres = origin + (idx + 0.5) * pitch

    # --- colours: z-buffer the blocks into the picture and average the pixels each one owns
    uv, depth = project(centres, R, fit)
    half = pitch * fit[0] * 0.62
    h, w = mask.shape
    zbuf = np.full((h, w), -np.inf)
    owner = np.full((h, w), -1, np.int64)
    for i in np.argsort(depth):  # far to near
        u, v = uv[i]
        x0, x1 = int(max(0, u - half)), int(min(w, u + half + 1))
        y0, y1 = int(max(0, v - half)), int(min(h, v + half + 1))
        if x0 >= x1 or y0 >= y1:
            continue
        sl = (slice(y0, y1), slice(x0, x1))
        nearer = depth[i] >= zbuf[sl]
        zbuf[sl][nearer] = depth[i]
        owner[sl][nearer] = i
    owner[~mask] = -1
    colours = np.full((len(idx), 3), np.nan)
    pix = {}
    for i in np.unique(owner[owner >= 0]):
        sel = owner == i
        pix[i] = rgba[..., :3][sel]
        colours[i] = np.median(pix[i], 0)
    visible = ~np.isnan(colours[:, 0])
    print(f"visible blocks {visible.sum()} / {len(idx)}")

    # hidden blocks: copy from the nearest visible block through the shell
    pos = {tuple(p): i for i, p in enumerate(idx)}
    q = deque(i for i in range(len(idx)) if visible[i])
    while q:
        i = q.popleft()
        for d in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
            j = pos.get(tuple(idx[i] + d))
            if j is not None and np.isnan(colours[j, 0]):
                colours[j] = colours[i]
                q.append(j)
    colours[np.isnan(colours[:, 0])] = 128

    # --- roles: which part of the build each block is
    face_tree = cKDTree(mesh.triangles_center)
    _, near = face_tree.query(centres)
    normals = mesh.face_normals[near]
    H = solid.shape[1]
    lab = to_lab(colours)
    green = (lab[:, 1] < -6) & (lab[:, 2] > 6)  # leaves / bushes / moss patches
    roles = np.empty(len(idx), object)
    for i, (bx, by, bz) in enumerate(idx):
        above_empty = by + 1 >= H or not solid[bx, by + 1, bz]
        n = normals[i]
        if green[i]:
            roles[i] = "plant"
        elif above_empty and n[1] > 0.3 and by >= H * 0.35:
            roles[i] = "roof"
        else:
            roles[i] = "wall"

    # --- materials per role: cluster colours, one block per cluster, from blocks that suit the role
    names, tex_mean, tex_std = texture_stats()
    names_arr = np.array(names)
    has_stairs = np.array([BLOCKS[n][1] is not None for n in names])
    is_leaf = np.array(["leaves" in n for n in names])
    is_wood = np.array([any(t in n for t in ("planks", "log")) for n in names])

    def stats_of(members):
        px = np.concatenate([pix[i] for i in members if i in pix] or [colours[members]])
        plab = to_lab(px)
        return plab.mean(0), plab[:, 0].std()

    def cost(shift, m, s_):
        return (np.abs(tex_mean[:, 1] - m[1]) + np.abs(tex_mean[:, 2] - m[2]))             + np.abs(tex_mean[:, 0] - (m[0] + shift)) * 0.6 + np.abs(tex_std - s_) * 0.3

    groups = []  # (role, member indices)
    for role, k in (("roof", 3), ("wall", 5)):
        members = np.where(roles == role)[0]
        if len(members) == 0:
            continue
        vis = [i for i in members if visible[i]] or list(members)
        _, cent = kmeans(lab[vis], min(k, len(vis)))
        a_ = np.argmin(((lab[members][:, None] - cent[None]) ** 2).sum(-1), 1)
        for j in range(len(cent)):
            if (a_ == j).any():
                groups.append((role, members[a_ == j]))
    stats = [stats_of(list(m)) for _, m in groups]
    best_shift = min(range(-10, 41, 2), key=lambda sh: sum(len(m) * cost(sh, st[0], st[1]).min()
                                                          for (_, m), st in zip(groups, stats)))
    print(f"brightness shift +{best_shift} L")
    block_of = {}
    for (role, members), (m, s_) in zip(groups, stats):
        c = cost(best_shift, m, s_)
        if role == "roof":
            c = np.where(has_stairs & ~is_leaf, c, np.inf)  # roofs need a stair version
        else:
            c = np.where(~is_leaf, c, np.inf)
        name = names_arr[int(np.argmin(c))]
        for i in members:
            block_of[i] = name
        print(f"  {role:5s} {len(members):4d} blocks -> {name}")
    for i in np.where(roles == "plant")[0]:
        block_of[i] = "azalea_leaves"
    print(f"  plant {int((roles == 'plant').sum()):4d} blocks -> azalea_leaves")

    # --- Minecraft parts: stairs on roof slopes, slabs on half-filled tops
    states = []
    for i, (bx, by, bz) in enumerate(idx):
        name = block_of[i]
        _, stairs, slab = BLOCKS[name]
        n = normals[i]
        state = "minecraft:" + name
        if "leaves" in name:
            state += "[persistent=true]"
        elif roles[i] == "roof" and stairs and np.hypot(n[0], n[2]) > 0.25 and n[1] < 0.95:
            # the slope's normal points downhill; a stair's tall back faces uphill
            facing = (("east" if n[0] < 0 else "west") if abs(n[0]) > abs(n[2])
                      else ("south" if n[2] < 0 else "north"))
            state = f"minecraft:{stairs}[facing={facing},half=bottom]"
        elif roles[i] == "roof" and slab and lower[bx, by, bz] > 0.6 and upper[bx, by, bz] < 0.3:
            state = f"minecraft:{slab}[type=bottom]"
        elif "log" in name:
            state += "[axis=y]"
        states.append(state)

    if dbg:
        img = Image.fromarray(rgba[..., :3].astype(np.uint8))
        d = ImageDraw.Draw(img)
        for i in range(len(idx)):
            if visible[i]:
                u, v = uv[i]
                d.rectangle([u - half, v - half, u + half, v + half], outline=(255, 0, 0))
        img.save(dbg / f"{a.name}_projection.png")

    # --- place on the ground and write the blueprint
    sys.path.insert(0, str(STUDIO))
    import studio
    from region import RegionReader, region_dir
    rr = RegionReader(region_dir(a.world, "minecraft_overworld"))
    not_ground = ("air", "leaves", "_log", "grass", "fern", "flower", "bush", "vine", "poppy", "dandelion",
                  "tulip", "daisy", "bluet", "orchid", "allium", "cornflower", "lily", "water", "snow")

    def ground_at(x, z):
        y = rr.surface(x, z)
        while y > -64 and any(t in rr.block(x, y, z) for t in not_ground):
            y -= 1
        return y

    gi = idx - idx.min(0)
    wx, wz = int(gi[:, 0].max() + 1), int(gi[:, 2].max() + 1)
    gx, gz = a.at
    x0, z0 = gx - wx // 2, gz - wz // 2
    ground = min(ground_at(x0 + dx, z0 + dz) for dx in range(0, wx, 2) for dz in range(0, wz, 2))
    out, taken = [], set()
    for (ix, iy, iz), s in zip(gi, states):
        x, y, z = x0 + int(ix), ground + 1 + int(iy), z0 + int(iz)
        out.append([x, y, z, s])
        taken.add((x, y, z))
    top = ground + 1 + int(gi[:, 1].max()) + 8
    for x in range(x0 - 3, x0 + wx + 3):
        for z in range(z0 - 3, z0 + wz + 3):
            for y in range(ground + 1, top):
                if (x, y, z) not in taken and any(t in rr.block(x, y, z) for t in ("leaves", "_log", "grass", "fern", "bush")):
                    out.append([x, y, z, "minecraft:air"])
    cfg = studio.load_config()
    res = studio.export_blueprint({"gameDir": cfg["gameDirs"][0], "worldId": "sp_" + studio.sanitize(Path(a.world).name),
                                   "dimension": "minecraft_overworld", "name": a.name, "blocks": out}, cfg)
    print("blueprint:", res)


if __name__ == "__main__":
    main()
