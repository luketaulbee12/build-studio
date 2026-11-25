"""Background removal without extra model downloads.

1. Find where the build is: Minecraft builds are full of texture, studio-style backgrounds are smooth,
   so the busy region gives a starting box.
2. Refine with OpenCV GrabCut (colour models for object vs background).
3. Keep the largest piece (drops watermarks/text), crop, pad to a square RGBA image.
"""
import cv2
import numpy as np
from PIL import Image
from skimage import measure, morphology


_SESSION = None


def _birefnet(img):
    global _SESSION
    import os
    os.environ.setdefault("U2NET_HOME", str(__import__("pathlib").Path(__file__).resolve().parent.parent / "models" / "rembg"))
    from rembg import new_session, remove
    if _SESSION is None:
        _SESSION = new_session("birefnet-general")
    return np.asarray(remove(img, session=_SESSION, only_mask=True)) > 127


def matte(path, pad_ratio=0.85, iters=6):
    """Cut-out is cached per picture (the slow AI step then runs once, not once per pipeline stage)."""
    import hashlib
    from pathlib import Path
    p = Path(path)
    key = hashlib.sha1(f"{p.resolve()}|{p.stat().st_mtime}|{pad_ratio}".encode()).hexdigest()[:16]
    cache = Path(__file__).resolve().parent.parent / "out" / "matte_cache" / f"{key}.png"
    if cache.exists():
        return Image.open(cache).convert("RGBA")
    im = _matte(path, pad_ratio, iters)
    cache.parent.mkdir(parents=True, exist_ok=True)
    im.save(cache)
    return im


def _matte(path, pad_ratio=0.85, iters=6):
    src = Image.open(path)
    if src.mode in ("RGBA", "LA", "P"):
        a = np.asarray(src.convert("RGBA"))[..., 3]
        if (a < 128).mean() > 0.05:  # already cut out (e.g. wiki renders): use its own transparency
            return _crop_pad(np.asarray(src.convert("RGB")), a > 127, pad_ratio)
    rgb = np.asarray(src.convert("RGB"))
    try:  # AI cut-out (BiRefNet): handles real photos with sky, trees and grass
        fg = _birefnet(src.convert("RGB"))
        if fg.mean() > 0.01:
            lab = measure.label(fg)
            fg = lab == (np.argmax(np.bincount(lab.ravel())[1:]) + 1)
            return _crop_pad(rgb, fg, pad_ratio)
    except Exception as e:  # never fall back silently: the colour-based cut ruins photos
        raise RuntimeError(f"AI background cut-out failed for {path}: {e}") from e
    raise RuntimeError(f"AI background cut-out found nothing in {path}")
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    h, w = rgb.shape[:2]

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    energy = cv2.GaussianBlur(np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3)), (0, 0), 9)
    busy = energy > np.percentile(energy, 80)
    lab = measure.label(busy)
    biggest = np.argmax(np.bincount(lab.ravel())[1:]) + 1
    ys, xs = np.where(lab == biggest)
    m = int(0.03 * max(h, w))
    x0, y0 = max(0, xs.min() - m), max(0, ys.min() - m)
    x1, y1 = min(w - 1, xs.max() + m), min(h - 1, ys.max() + m)

    mask = np.zeros((h, w), np.uint8)
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    cv2.grabCut(bgr, mask, (int(x0), int(y0), int(x1 - x0), int(y1 - y0)), bgd, fgd, iters, cv2.GC_INIT_WITH_RECT)
    fg = (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)
    fg = morphology.opening(fg, morphology.disk(2))
    lab = measure.label(fg)
    if lab.max() == 0:
        raise ValueError("couldn't find the build in the picture")
    biggest = np.argmax(np.bincount(lab.ravel())[1:]) + 1
    fg = morphology.closing(lab == biggest, morphology.disk(2))
    fg = morphology.remove_small_holes(fg, 400)
    return _crop_pad(rgb, fg, pad_ratio)


def _crop_pad(rgb, fg, pad_ratio):
    ys, xs = np.where(fg)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    rgba = np.dstack([rgb, fg.astype(np.uint8) * 255])[y0:y1, x0:x1]
    size = int(max(y1 - y0, x1 - x0) / pad_ratio)
    out = np.zeros((size, size, 4), np.uint8)
    oy, ox = (size - (y1 - y0)) // 2, (size - (x1 - x0)) // 2
    out[oy:oy + y1 - y0, ox:ox + x1 - x0] = rgba
    return Image.fromarray(out, "RGBA")


if __name__ == "__main__":
    import sys
    im = matte(sys.argv[1])
    im.save(sys.argv[2])
    print("saved", sys.argv[2], im.size)
