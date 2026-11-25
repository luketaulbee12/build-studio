"""Run Hunyuan3D-2 mini (shape) and/or TripoSR on one picture; save meshes to out/.

usage: python scripts/run_models.py <image> <name> [hunyuan|triposr|both]
"""
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts" / "shim"))
sys.path.insert(0, str(ROOT / "scripts"))
from matte import matte  # noqa: E402

OUT = ROOT / "out"
OUT.mkdir(exist_ok=True)


def run_hunyuan(rgba, name):
    os.environ["HY3DGEN_MODELS"] = str(ROOT / "models")
    sys.path.insert(0, str(ROOT / "Hunyuan3D-2"))
    from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
    t = time.time()
    pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        "tencent/Hunyuan3D-2mini", subfolder="hunyuan3d-dit-v2-mini", use_safetensors=True, variant="fp16")
    print(f"hunyuan loaded in {time.time() - t:.1f}s")
    t = time.time()
    mesh = pipe(image=rgba, num_inference_steps=30, octree_resolution=256, num_chunks=20000,
                generator=torch.manual_seed(1234), output_type="trimesh")[0]
    print(f"hunyuan generated in {time.time() - t:.1f}s, {len(mesh.vertices)} verts; "
          f"peak VRAM {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")
    f = OUT / f"{name}_hunyuan.glb"
    mesh.export(f)
    del pipe
    torch.cuda.empty_cache()
    return f


def run_triposr(rgba, name):
    sys.path.insert(0, str(ROOT / "TripoSR"))
    from tsr.system import TSR
    t = time.time()
    model = TSR.from_pretrained(str(ROOT / "models" / "stabilityai" / "TripoSR"),
                                config_name="config.yaml", weight_name="model.ckpt")
    model.renderer.set_chunk_size(8192)
    model.to("cuda")
    print(f"triposr loaded in {time.time() - t:.1f}s")
    img = np.asarray(rgba).astype(np.float32) / 255.0
    img = img[:, :, :3] * img[:, :, 3:4] + (1 - img[:, :, 3:4]) * 0.5
    img = Image.fromarray((img * 255).astype(np.uint8)).resize((512, 512))
    t = time.time()
    with torch.no_grad():
        codes = model([img], device="cuda")
    mesh = model.extract_mesh(codes, True, resolution=256)[0]
    print(f"triposr generated in {time.time() - t:.1f}s, {len(mesh.vertices)} verts; "
          f"peak VRAM {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")
    f = OUT / f"{name}_triposr.ply"
    mesh.export(f)
    return f


if __name__ == "__main__":
    image, name = sys.argv[1], sys.argv[2]
    which = sys.argv[3] if len(sys.argv) > 3 else "both"
    rgba = matte(image)
    rgba.save(OUT / f"{name}_matte.png")
    if which in ("hunyuan", "both"):
        print("saved", run_hunyuan(rgba, name))
    if which in ("triposr", "both"):
        print("saved", run_triposr(rgba, name))
