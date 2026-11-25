"""Texture a Hunyuan shape with Hunyuan3D-2 Paint (all sides, lighting removed from the picture).

usage: python scripts/paint.py <shape.glb> <picture> <out.glb>
"""
import os
import sys
import time
from pathlib import Path

import torch
import trimesh

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "Hunyuan3D-2"))
from matte import matte  # noqa: E402


def main(shape, picture, out):
    from hy3dgen.shapegen.postprocessors import FaceReducer, FloaterRemover, DegenerateFaceRemover
    from hy3dgen.texgen import Hunyuan3DPaintPipeline

    mesh = trimesh.load(shape, force="mesh")
    mesh = FloaterRemover()(mesh)
    mesh = DegenerateFaceRemover()(mesh)
    mesh = FaceReducer()(mesh, max_facenum=40000)
    image = matte(picture)

    t = time.time()
    pipe = Hunyuan3DPaintPipeline.from_pretrained(str(ROOT / "models" / "tencent" / "Hunyuan3D-2"),
                                                  subfolder="hunyuan3d-paint-v2-0-turbo")
    pipe.enable_model_cpu_offload()  # keeps it inside 12 GB
    print(f"paint loaded in {time.time() - t:.1f}s")
    t = time.time()
    textured = pipe(mesh, image=image)
    print(f"painted in {time.time() - t:.1f}s; peak VRAM {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")
    textured.export(out)
    print("saved", out)


if __name__ == "__main__":
    main(*sys.argv[1:4])
