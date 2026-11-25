# Build Studio

Design Minecraft builds outside the game, preview them in 3D with real block textures, and send them to the
**freecam-ghost** mod as ghost-block blueprints (open them in game with **J**). Includes a local AI pipeline that
turns an inspiration picture (a photo or a Minecraft screenshot) into a blueprint.

Everything runs on your own PC - no paid API.

## Parts
| Folder | What it is |
|---|---|
| `studio/` | The Build Studio web app (Python stdlib server + three.js viewer). Reads your world's region files for terrain, has a style-based designer, a Superflat preview, and "Build from picture". |
| `scripts/` | The picture -> build pipeline (below). `pipeline.py` runs all of it. |
| `data/` | Block palettes builders use (for texture variety) and the test-set list. |

## Running the studio
```
cd studio
python studio.py          # then open http://localhost:8765
```
First run extracts block models/textures from your Minecraft 1.21.1 client jar. Settings go in
`studio/config.json` (created on first run): `clientJar`, `gameDirs` (game folders that hold `freecam_ghosts/`
and `saves/`), `worlds` (extra world folders, e.g. your server's).

## Picture -> build pipeline (optional, needs an NVIDIA GPU, ~12 GB VRAM)
1. `matte.py` - cut the building out of the picture (BiRefNet via rembg).
2. `run_models.py` - 3D shape (Hunyuan3D-2mini).
3. `paint.py` - texture every side (Hunyuan3D-2 Paint, turbo, ~7 GB VRAM).
4. `build_from_model.py` - blocks at true scale:
   - line walls up with the block grid, rebuild straight walls and roof planes as stairs/slabs (`regularize.py`, `planes2blocks.py`);
   - partial blocks per cell (`shapes.py`);
   - CLIP names what walls / roof are made of (`material_family.py`), one material per face (`materials.py`);
   - Minecraft screenshots use the picture's own colours;
   - windows, doors, corner/base trim (`detail.py`); related-block variety from builder palettes (`variety.py`).

```
.venv/Scripts/python.exe scripts/pipeline.py <picture> "<name>" --width 16 --world-id sp_<World_Name>
```
or use **Build from picture** in the studio.

### Setup
- Python 3.12 venv in `.venv` with torch (CUDA), transformers<5, diffusers<0.36, trimesh, pymeshlab, scipy,
  scikit-image, opencv, rembg, onnxruntime.
- Clone Tencent's `Hunyuan3D-2` repo into `Hunyuan3D-2/`; build Paint's native add-ons with `scripts/build_paint_ext.bat`.
- Models go in `models/` (not in git): `tencent/Hunyuan3D-2mini`, `tencent/Hunyuan3D-2` (delight + paint-turbo,
  fp16 safetensors - apply `patches/hunyuan_paint_fp16_safetensors.diff` to both copies of `unet/modules.py`),
  `clip-vit-base-patch16`, `clip-vit-large-patch14`, `rembg/` (birefnet-general).
- Hunyuan3D models are under the Tencent Hunyuan Community License.

## Quality testing
`scripts/run_testset.py` builds every picture in `data/testset.json` and writes side-by-side sheets to
`out/testset/`; changes are kept only if the whole set improves. Progress notes: `STATUS.md`.
