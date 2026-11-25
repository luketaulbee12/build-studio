# Status (autonomous session, 2026-11-24)

How changes are judged: every change runs on all 16 test pictures (`data/testset.json`), and is kept only if
the set as a whole looks better. Sheets are in `out/testset/*_cmp.jpg` (picture | before | after).

## Pipeline now
picture -> BiRefNet cut-out -> Hunyuan shape -> Hunyuan Paint -> `build_from_model.py`:
1. Line walls up with the block grid, flatten surfaces onto planes (`regularize.py`).
2. Shapes per block (stairs/slabs/...), then **rebuild from planes** (`planes2blocks.py`): straight walls,
   roofs as exact stair/slab rows.
3. Materials (`materials.py`): CLIP names what walls/roof are made of; **one material per wall/roof face**;
   a second material only for a large solid region that differs in colour (not shade); big dark areas are
   shadow, not windows; shaded faces use the building's sunlit material.
4. **Detailing** (`detail.py`): windows on walls the photo didn't see, a doorway, corner + base trim.
5. Variety from 3000 blockpalettes.com palettes (`variety.py`); decorative blocks never random.

## Results so far (16 buildings)
| Change | Better | Same | Worse |
|---|---|---|---|
| Rebuild from planes | ~10 | 5 | 1 (barn's thin roof) |
| One material per face | 9 | 4 | 3 (fixed 2 of 3 after) |
| Detailing | most houses (stone base, log corners, windows) | | Kisimul got wood trim (fixing) |
| Roof label (whole-photo + roof crop) | 7/11 roofs right vs 6/11; thatched roofs now brown not grey | | A-frame (metal) now reads as wood |
| Masonry trim rule (no log corners on stone) | Kisimul | rest | |
| Roof slopes: material without stairs -> closest block with stairs | thatch roofs keep stair shape | rest | |
| Photo colour per visible face (experiment) | none clearly | | ~6 (darker/brown, noisy) -> rejected, off |
| Snap face colours to the picture's own palette | Minecraft screenshots (village houses get grey stone walls) | | real photos (lighting in the palette) |
| -> only for Minecraft screenshots (CLIP: screenshots >= 0.99, photos <= 0.25) | screenshots | photos unchanged | |
| Walls face the real outside (flood fill) + real doors | big village house gets windows on its blank wall, medium house door + windows | most | Kisimul: back-wall windows read as dark speckles on a castle |
| Corner trim only where two real walls meet (each runs >= 3 blocks) | Kisimul (speckles gone), castle keep, round tower | houses | |
| Bridge one-block gaps in thin sheets (on) | barn, tower slightly more complete | rest | |
| Generated windows match the photo's window density (option `--windens 1`) | | all (no visible change) | -> off |
| Even out ragged tops (option `--tops 1`) | farmhouse ridge, tower top a bit | most | cottage loses some plants -> off |

## Known problems, ranked by how many buildings they affect
1. Paint paints small mixed materials as one colour. Fixed for Minecraft screenshots (palette snapping);
   still open for real photos.
2. Roof material: large CLIP (ViT-L/14) for roofs = 9/11 right (was 7/11): A-frame now metal, farmhouse slate.
   Walls stay on the small model (large misread sandstone walls as grey). Misses: Mattagua roof (slate -> wood),
   brick cottage (clay -> slate).
3. Ragged wall tops / battlements (3 builds).
4. Thin structures (barn roof, A-frame) break apart - gap bridging helps a little; still the weakest shape case.
5. Castle keep regressed when the colour-spread step came back (greenery in photo).

## Real-build corpus (for learning how builders build)
`scripts/corpus.py` loads vanilla structure .nbt, .litematic, .schem and CraftAssist (1.12 ids via
`legacy_blocks.py`) into one format (`corpus/builds/*.npz`, not in git). Sources fetched into `corpus/raw/`:
the game jar's 1,180 structures, Hack337/Minecraft-Schematics, CraftAssist houses (~2,500); more HF sets were
rate-limited (retry needs an HF login/token). Result: 3,825 builds, 2.87M blocks. Next: mine local patterns
(corners, roof edges, window frames, trims) to replace hand-written detailing rules.

## Build Studio
"Build from picture" button (Inspiration section): runs this whole pipeline in the background (~2.5 min),
then shows the blueprint (on the picked plot, or the Superflat preview) and it's in J in game.

## Not done / needs you
- Nothing pushed; local commits only in `recon` (and freecam-ghost earlier).
- Bigger CLIP download (item 2) waits for your OK.
