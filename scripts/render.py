"""Render a block build from the picture's camera and score it against the picture.

Flat-shaded faces (each block's real texture colour), orthographic, painter's algorithm. Good enough to
tell whether a change made the build look more or less like the picture.
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from convert import project, to_lab

FACES = {  # normal -> 4 corner offsets (unit cube)
    (1, 0, 0): [(1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1)],
    (-1, 0, 0): [(0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0)],
    (0, 1, 0): [(0, 1, 0), (0, 1, 1), (1, 1, 1), (1, 1, 0)],
    (0, -1, 0): [(0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1)],
    (0, 0, 1): [(0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)],
    (0, 0, -1): [(0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0)],
}
LIGHT = np.array([-0.35, 1.0, 0.55]) / np.linalg.norm([-0.35, 1.0, 0.55])


def faces(cells, R, fit, origin, pitch):
    """Camera-facing exposed faces, far first: (depth, pixel polygon, cell, normal)."""
    out = []
    occupied = set(cells)
    for (x, y, z) in cells:
        for n, corners in FACES.items():
            if (x + n[0], y + n[1], z + n[2]) in occupied:
                continue
            if (R @ np.array(n, float))[2] <= 1e-6:  # facing away from the camera
                continue
            pts = origin + (np.array([x, y, z]) + np.array(corners)) * pitch
            uv, d = project(pts, R, fit)
            out.append((d.mean(), [tuple(p) for p in uv], (x, y, z), n))
    out.sort(key=lambda p: p[0])
    return out


def render(cells, colour_of, R, fit, shape, origin, pitch, bg=(0, 0, 0)):
    """cells: {(ix,iy,iz): name}; colour_of: name -> RGB. Returns (rgb array, mask)."""
    h, w = shape
    img = Image.new("RGB", (w, h), bg)
    m = Image.new("L", (w, h), 0)
    di, dm = ImageDraw.Draw(img), ImageDraw.Draw(m)
    for _, pts, cell, n in faces(cells, R, fit, origin, pitch):
        shade = 0.55 + 0.45 * max(0.0, float(np.dot(n, LIGHT)))
        col = tuple(int(c) for c in np.clip(np.asarray(colour_of(cells[cell]), float) * shade, 0, 255))
        di.polygon(pts, fill=col)
        dm.polygon(pts, fill=255)
    return np.asarray(img), np.asarray(m) > 127


def owners(cells, R, fit, shape, origin, pitch):
    """Which cell each picture pixel shows (index into list(cells)), -1 for none."""
    h, w = shape
    keys = list(cells)
    index = {k: i for i, k in enumerate(keys)}
    img = Image.new("I", (w, h), -1)
    nimg = Image.new("L", (w, h), 255)
    d, dn = ImageDraw.Draw(img), ImageDraw.Draw(nimg)
    dirs = list(FACES)
    for _, pts, cell, n in faces(cells, R, fit, origin, pitch):
        d.polygon(pts, fill=index[cell])
        dn.polygon(pts, fill=dirs.index(n))
    nm = np.asarray(nimg, np.int64)
    nm[nm == 255] = -1
    return np.asarray(img, np.int64), keys, nm  # nm: index into list(FACES) of the face each pixel shows


def score(pic_rgb, pic_mask, ren_rgb, ren_mask, block_px):
    """(silhouette IoU, mean colour error in Lab after blurring to block scale). Lower error = closer."""
    iou = (pic_mask & ren_mask).sum() / max(1, (pic_mask | ren_mask).sum())
    r = float(max(1, block_px / 2))
    a = to_lab(np.asarray(Image.fromarray(pic_rgb.astype(np.uint8)).filter(ImageFilter.GaussianBlur(r)), float))
    b = to_lab(np.asarray(Image.fromarray(ren_rgb.astype(np.uint8)).filter(ImageFilter.GaussianBlur(r)), float))
    both = pic_mask & ren_mask
    if not both.any():
        return iou, 99.0
    d = a[both] - b[both]
    d[:, 0] -= np.median(d[:, 0])  # ignore overall exposure differences
    return iou, float(np.sqrt((d ** 2).sum(1)).mean())


def side_by_side(pic_rgb, ren_rgb, path):
    h, w = pic_rgb.shape[:2]
    out = np.zeros((h, w * 2 + 8, 3), np.uint8)
    out[:, :w] = pic_rgb
    out[:, w + 8:] = ren_rgb
    Image.fromarray(out).save(path)


# ------------------------------------------------------------------------------- shaped blocks (stairs, slabs ...)

def _boxes(shape):
    """Unit-cell boxes (min, max) for a block shape name from shapes.py."""
    if shape in (None, "full"):
        return [((0, 0, 0), (1, 1, 1))]
    if shape == "slab_bottom":
        return [((0, 0, 0), (1, .5, 1))]
    if shape == "slab_top":
        return [((0, .5, 0), (1, 1, 1))]
    if shape == "floor":
        return [((0, 0, 0), (1, .19, 1))]
    if shape == "post":
        return [((.375, 0, .375), (.625, 1, .625))]
    if shape == "pane_x":
        return [((0, 0, .44), (1, 1, .56))]
    if shape == "pane_z":
        return [((.44, 0, 0), (.56, 1, 1))]
    side = {"north": ((0, 0, 0), (1, 1, .19)), "south": ((0, 0, .81), (1, 1, 1)),
            "west": ((0, 0, 0), (.19, 1, 1)), "east": ((.81, 0, 0), (1, 1, 1))}
    if shape.startswith("plate"):
        return [side[shape.split("_")[1]]]
    if shape.startswith("stairs"):
        _, f, half = shape.split("_")
        lo_, hi_ = ((0, 0, 0), (1, .5, 1)) if half == "bottom" else ((0, .5, 0), (1, 1, 1))
        back = {"north": ((0, 0, 0), (1, 1, .5)), "south": ((0, 0, .5), (1, 1, 1)),
                "west": ((0, 0, 0), (.5, 1, 1)), "east": ((.5, 0, 0), (1, 1, 1))}[f]
        y0, y1 = (.5, 1) if half == "bottom" else (0, .5)
        b = ((back[0][0], y0, back[0][2]), (back[1][0], y1, back[1][2]))
        return [(lo_, hi_), b]
    return [((0, 0, 0), (1, 1, 1))]


def render_shapes(cells, colour_of, R, fit, shape_hw, origin, pitch, bg=(0, 0, 0)):
    """cells: {(x,y,z): (material, shape)}. Draws real block shapes; hidden faces of full blocks are skipped."""
    h, w = shape_hw
    full = {p for p, (_, s) in cells.items() if s in (None, "full")}
    polys = []
    for (x, y, z), (mat, shp) in cells.items():
        col = np.asarray(colour_of(mat), float)
        for lo, hi in _boxes(shp):
            lo, hi = np.array(lo, float), np.array(hi, float)
            for n, corners in FACES.items():
                if shp in (None, "full") and (x + n[0], y + n[1], z + n[2]) in full:
                    continue
                if (R @ np.array(n, float))[2] <= 1e-6:
                    continue
                c = np.array(corners, float)
                pts = origin + (np.array([x, y, z]) + lo + c * (hi - lo)) * pitch
                uv, d = project(pts, R, fit)
                shade = 0.55 + 0.45 * max(0.0, float(np.dot(n, LIGHT)))
                polys.append((d.mean(), [tuple(p) for p in uv], tuple(int(v) for v in np.clip(col * shade, 0, 255))))
    polys.sort(key=lambda p: p[0])
    img = Image.new("RGB", (w, h), bg)
    m = Image.new("L", (w, h), 0)
    di, dm = ImageDraw.Draw(img), ImageDraw.Draw(m)
    for _, pts, col in polys:
        di.polygon(pts, fill=col, outline=tuple(max(0, c - 25) for c in col))
        dm.polygon(pts, fill=255)
    return np.asarray(img), np.asarray(m) > 127
