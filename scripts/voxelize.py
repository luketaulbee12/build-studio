"""Turn a reconstructed 3D model into Minecraft blocks and drop it into the studio/test world as a blueprint.

usage: python scripts/voxelize.py <mesh> <matte.png> <name> --width 13 --at X Z [--world <path>]

Steps:
  1. Find which side of the model is the "front" (the side the picture was taken from) by comparing each
     side's silhouette with the picture's cut-out.
  2. Scale so the front is --width blocks wide, voxelize, keep a 1-block shell (hollow inside).
  3. Colour each block from the model's vertex colours (TripoSR) or by projecting the picture (Hunyuan).
  4. Pick the Minecraft block whose real texture colour is closest (perceptual Lab distance).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image
from scipy.spatial import cKDTree

STUDIO = Path(__file__).resolve().parent.parent / "studio"  # Build Studio lives next to scripts/
TEX = STUDIO / "data" / "mc" / "textures" / "block"

# Full-cube blocks a builder would plausibly use (texture file -> block id).
CANDIDATES = """
stone cobblestone mossy_cobblestone stone_bricks mossy_stone_bricks cracked_stone_bricks andesite polished_andesite
diorite polished_diorite granite polished_granite tuff calcite deepslate_bricks cobbled_deepslate deepslate_tiles
bricks mud_bricks packed_mud sandstone smooth_stone gravel dirt coarse_dirt
oak_planks spruce_planks birch_planks dark_oak_planks jungle_planks acacia_planks mangrove_planks cherry_planks
stripped_oak_log stripped_spruce_log stripped_dark_oak_log stripped_birch_log
white_terracotta terracotta light_gray_terracotta brown_terracotta gray_terracotta
white_concrete light_gray_concrete gray_concrete
oak_leaves spruce_leaves azalea_leaves moss_block
barrel_side glass
""".split()


def srgb_to_lab(c):
    c = np.asarray(c, np.float64) / 255.0
    c = np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)
    m = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
    xyz = c @ m.T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def block_palette():
    names, cols = [], []
    for n in CANDIDATES:
        f = TEX / f"{n}.png"
        if not f.exists():
            continue
        px = np.asarray(Image.open(f).convert("RGBA")).astype(np.float64)[:16]
        a = px[..., 3] > 0
        col = px[..., :3][a].mean(0)
        if "leaves" in n:  # leaves are grey textures tinted green in game
            col = col * np.array([0.47, 0.67, 0.18]) / 0.6
        names.append(n.replace("_side", ""))
        cols.append(col)
    return names, srgb_to_lab(np.array(cols))


def best_front(mesh, alpha):
    """Rotation (about Y) whose front silhouette best matches the picture's cut-out."""
    mask = np.asarray(alpha) > 127
    ys, xs = np.where(mask)
    mask = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    best = None
    for k in range(4):
        R = trimesh.transformations.rotation_matrix(k * np.pi / 2, [0, 1, 0])
        v = trimesh.transform_points(mesh.vertices, R)
        for flip in (1, -1):
            # look from +Z toward -Z: image x = vertex x, image y = -vertex y
            sil = silhouette(v * np.array([flip, 1, 1]), mesh.faces, mask.shape)
            iou = (sil & mask).sum() / max(1, (sil | mask).sum())
            if best is None or iou > best[0]:
                best = (iou, k, flip)
    return best


def silhouette(v, faces, shape):
    from PIL import ImageDraw
    h, w = shape
    x, y = v[:, 0], -v[:, 1]
    x = (x - x.min()) / max(1e-9, x.max() - x.min()) * (w - 1)
    y = (y - y.min()) / max(1e-9, y.max() - y.min()) * (h - 1)
    img = Image.new("1", (w, h), 0)
    d = ImageDraw.Draw(img)
    for f in faces[:: max(1, len(faces) // 60000)]:
        d.polygon([(x[i], y[i]) for i in f], fill=1)
    return np.asarray(img, bool)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mesh")
    ap.add_argument("matte")
    ap.add_argument("name")
    ap.add_argument("--width", type=int, default=13)
    ap.add_argument("--at", type=int, nargs=2, required=True, metavar=("X", "Z"))
    ap.add_argument("--world", default=None, help="world folder to read the ground from (with --at); without it the build sits on --ground")
    a = ap.parse_args()

    scene = trimesh.load(a.mesh)
    mesh = scene.dump(concatenate=True) if isinstance(scene, trimesh.Scene) else scene
    matte = Image.open(a.matte).convert("RGBA")
    alpha = matte.getchannel("A")

    iou, k, flip = best_front(mesh, alpha)
    print(f"front view match: IoU {iou:.2f} (rotation {k * 90}°, mirror {flip < 0})")
    R = trimesh.transformations.rotation_matrix(k * np.pi / 2, [0, 1, 0])
    mesh.apply_transform(R)
    if flip < 0:
        mesh.apply_scale([-1, 1, 1])
        mesh.invert()

    lo, hi = mesh.bounds
    pitch = (hi[0] - lo[0]) / a.width
    vox = mesh.voxelized(pitch).fill()
    occ = vox.matrix.copy()  # indices: [x, y, z] in the voxel grid
    # 1-block shell: keep voxels with at least one empty neighbour
    pad = np.pad(occ, 1)
    inner = pad[2:, 1:-1, 1:-1] & pad[:-2, 1:-1, 1:-1] & pad[1:-1, 2:, 1:-1] & pad[1:-1, :-2, 1:-1] \
        & pad[1:-1, 1:-1, 2:] & pad[1:-1, 1:-1, :-2]
    shell = occ & ~inner
    idx = np.argwhere(shell)
    centres = vox.indices_to_points(idx)
    print(f"voxels: grid {occ.shape}, shell {len(idx)} blocks")

    # colours
    vc = getattr(mesh.visual, "vertex_colors", None)
    if vc is not None and len(vc) == len(mesh.vertices) and np.ptp(np.asarray(vc)[:, :3]) > 0:
        tree = cKDTree(mesh.vertices)
        _, nn = tree.query(centres, k=4)
        cols = np.asarray(vc)[:, :3][nn].mean(1)
        source = "model colours"
    else:
        # project the picture from the front (orthographic): picture x <-> model x, picture y <-> -model y
        rgba = np.asarray(matte).astype(np.float64)
        ys, xs = np.where(rgba[..., 3] > 127)
        x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
        u = (centres[:, 0] - lo[0]) / (hi[0] - lo[0]) * (x1 - x0) + x0
        v = (hi[1] - centres[:, 1]) / (hi[1] - lo[1]) * (y1 - y0) + y0
        cols = np.zeros((len(centres), 3))
        r = max(1, int((x1 - x0) / a.width / 3))
        for i, (uu, vv) in enumerate(zip(u.astype(int), v.astype(int))):
            patch = rgba[max(0, vv - r):vv + r + 1, max(0, uu - r):uu + r + 1]
            m = patch[..., 3] > 127
            cols[i] = patch[..., :3][m].mean(0) if m.any() else (128, 128, 128)
        source = "projected picture"
    names, pal = block_palette()
    lab = srgb_to_lab(cols)
    choice = np.argmin(((lab[:, None, :] - pal[None, :, :]) ** 2).sum(-1), 1)
    print(f"colours from {source}; blocks used:",
          {names[i]: int(c) for i, c in zip(*np.unique(choice, return_counts=True))})

    # place: bottom of the model on the ground at (X, Z), front facing south (+Z)
    sys.path.insert(0, str(STUDIO))
    import studio
    from region import RegionReader, region_dir
    rr = RegionReader(region_dir(a.world, "minecraft_overworld"))
    gx, gz = a.at
    not_ground = ("air", "leaves", "_log", "grass", "fern", "flower", "bush", "vine", "poppy", "dandelion",
                  "tulip", "daisy", "bluet", "orchid", "allium", "cornflower", "lily", "water", "snow")

    def ground_at(x, z):
        y = rr.surface(x, z)
        while y > -64 and any(t in rr.block(x, y, z) for t in not_ground):
            y -= 1
        return y

    gi = idx - idx.min(0)
    wx, wz = gi[:, 0].max() + 1, gi[:, 2].max() + 1
    x0, z0 = gx - wx // 2, gz - wz // 2
    ground = min(ground_at(x0 + dx, z0 + dz) for dx in range(0, wx, 2) for dz in range(0, wz, 2))
    blocks = []
    taken = set()
    for (ix, iy, iz), c in zip(gi, choice):
        x, y, z = x0 + int(ix), ground + 1 + int(iy), z0 + int(iz)
        blocks.append([x, y, z, "minecraft:" + names[c]])
        taken.add((x, y, z))
    # clear trees and plants around it so it can be seen (and built)
    top = ground + 1 + gi[:, 1].max() + 8
    for x in range(x0 - 3, x0 + wx + 3):
        for z in range(z0 - 3, z0 + wz + 3):
            for y in range(ground + 1, top):
                if (x, y, z) not in taken and any(t in rr.block(x, y, z) for t in ("leaves", "_log", "grass", "fern", "bush")):
                    blocks.append([x, y, z, "minecraft:air"])
    cfg = studio.load_config()
    world_id = "sp_" + studio.sanitize(Path(a.world).name)
    res = studio.export_blueprint({"gameDir": cfg["gameDirs"][0], "worldId": world_id,
                                   "dimension": "minecraft_overworld", "name": a.name, "blocks": blocks}, cfg)
    print("blueprint:", res)


if __name__ == "__main__":
    main()
