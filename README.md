# recon: picture -> Minecraft build

Runs locally (RTX 4070, 12 GB). One command: `.venv/Scripts/python.exe scripts/pipeline.py <picture> <name> --width <blocks> [--at X Z]`
(or `--all <folder>`). Results: `out/dbg/<name>_compare.png`, blueprint into the dev world; view with the
Build Studio "Superflat preview".

Pipeline
1. `matte.py` - cut the building out (BiRefNet via rembg; uses the picture's own transparency when it has one). Cached.
2. `run_models.py` - Hunyuan3D-2mini shape.
3. `paint.py` - Hunyuan3D-2 Paint (turbo) textures every side, lighting removed. ~7 GB VRAM.
4. `build_from_model.py` - blocks at true scale:
   - `shapes.py` partial-block shape matching (stairs, slabs, fences/walls, trapdoors, panes);
   - `material_family.py` CLIP names what walls / roof are made of;
   - `materials.py` one main block per part + accents, sun/shade evened, colour calibration;
   - `variety.py` related-block mixing learned from 3000 blockpalettes.com palettes (`data/`).

Setup notes
- Models (not in git): `models/tencent/Hunyuan3D-2mini`, `models/tencent/Hunyuan3D-2` (delight + paint-turbo,
  fp16 safetensors only), `models/clip-vit-base-patch16`, `models/rembg` (birefnet-general).
- Paint loads the fp16 safetensors via `patches/hunyuan_paint_fp16_safetensors.diff`; apply the same change to
  `models/tencent/Hunyuan3D-2/hunyuan3d-paint-v2-0-turbo/unet/modules.py` (diffusers loads that copy).
- Paint's C++/CUDA add-ons: `scripts/build_paint_ext.bat` (VS 2022 Build Tools + CUDA 12.9).
- Older experiments kept for reference: `convert.py`, `convert2.py`, `voxelize.py`, `voxelize_tex.py`.
