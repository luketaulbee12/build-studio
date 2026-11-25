"""What is each part of the building made of? (CLIP zero-shot on the part's pixels in the picture.)

Colour can't tell honey sandstone from birch planks, or grey slate from grey-painted wood; a vision model can.
The family then limits which blocks the part's MAIN material may be (accents stay free).
"""
from pathlib import Path

import numpy as np
from PIL import Image

MODEL = Path(__file__).resolve().parent.parent / "models" / "clip-vit-base-patch16"

STONE = ["cobblestone", "mossy_cobblestone", "stone_bricks", "mossy_stone_bricks", "cracked_stone_bricks", "andesite",
         "polished_andesite", "stone", "smooth_stone", "tuff", "tuff_bricks", "cobbled_deepslate", "deepslate_bricks",
         "deepslate_tiles", "polished_deepslate", "diorite", "calcite", "blackstone", "polished_blackstone_bricks"]
PALE_STONE = ["sandstone", "smooth_sandstone", "cut_sandstone", "end_stone_bricks", "diorite", "calcite",
              "polished_diorite", "stone_bricks", "andesite", "polished_andesite", "smooth_stone"]
BRICK = ["bricks", "mud_bricks", "granite", "polished_granite", "red_sandstone", "cut_red_sandstone", "terracotta",
         "nether_bricks"]
WOOD = ["oak_planks", "spruce_planks", "birch_planks", "dark_oak_planks", "jungle_planks", "acacia_planks",
        "stripped_oak_log", "stripped_spruce_log", "stripped_dark_oak_log", "stripped_birch_log", "oak_log",
        "spruce_log", "birch_log", "dark_oak_log"]
PLASTER = ["white_terracotta", "calcite", "white_concrete", "diorite", "polished_diorite", "quartz_block",
           "smooth_sandstone", "light_gray_terracotta", "birch_planks"]

BODY = {  # family -> (prompts, blocks)
    "grey stone": (["a wall made of grey stone", "a rough cobblestone wall", "a dark slate stone wall"], STONE),
    "pale stone": (["a wall made of pale yellow limestone blocks", "a sandstone castle wall"], PALE_STONE),
    "brick": (["a red brick wall", "a wall of orange clay bricks"], BRICK),
    "wood": (["a wall made of wooden planks", "a log cabin wall", "a wooden barn wall"], WOOD),
    "plaster": (["a white plaster wall", "a whitewashed stucco wall", "a cream painted render wall"], PLASTER),
}
ROOF = {
    "slate": (["a dark grey slate tile roof", "a stone shingle roof"],
              ["deepslate_tiles", "deepslate_bricks", "cobbled_deepslate", "polished_deepslate", "blackstone",
               "tuff_bricks", "stone_bricks", "cobblestone", "andesite", "tuff"]),
    "wood shingles": (["a wooden shingle roof", "a roof made of wooden planks"], WOOD),
    "thatch": (["a thatched straw roof", "a roof of dry reeds"],
               ["hay_block", "stripped_oak_log", "oak_planks", "birch_planks", "stripped_birch_log", "dripstone_block",
                "mud_bricks", "spruce_planks"]),
    "metal": (["a corrugated metal roof", "a grey tin roof"],
              ["smooth_stone", "polished_andesite", "andesite", "stone", "calcite", "diorite"]),
    "clay tiles": (["a red clay tile roof", "an orange terracotta tiled roof"],
                   ["bricks", "granite", "polished_granite", "terracotta", "mud_bricks", "red_sandstone",
                    "acacia_planks"]),
}

_model = None


def _load():
    global _model
    if _model is None:
        import torch
        from transformers import CLIPModel, CLIPProcessor
        m = CLIPModel.from_pretrained(str(MODEL)).eval().to("cuda" if torch.cuda.is_available() else "cpu")
        _model = (m, CLIPProcessor.from_pretrained(str(MODEL)))
    return _model


def classify(rgb, mask, families):
    """rgb: HxWx3 uint8 picture, mask: pixels of the part. Returns (best family, {family: probability})."""
    import torch
    ys, xs = np.where(mask)
    if len(ys) < 400:
        return None, {}
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    crop = rgb[y0:y1, x0:x1].copy()
    crop[~mask[y0:y1, x0:x1]] = 128
    img = Image.fromarray(crop.astype(np.uint8))
    m, proc = _load()
    texts, owner = [], []
    for fam, (prompts, _) in families.items():
        texts += prompts
        owner += [fam] * len(prompts)
    with torch.no_grad():
        inp = proc(text=texts, images=img, return_tensors="pt", padding=True).to(m.device)
        p = m(**inp).logits_per_image.softmax(-1)[0].cpu().numpy()
    score = {}
    for fam, pr in zip(owner, p):
        score[fam] = score.get(fam, 0) + float(pr)
    best = max(score, key=score.get)
    return best, score


def allowed_blocks(families, score, min_share=0.2):
    """Blocks of the winning family, plus a runner-up family that is clearly present too (stone + timber)."""
    ranked = sorted(score.items(), key=lambda t: -t[1])
    out = list(families[ranked[0][0]][1])
    mixed = len(ranked) > 1 and ranked[1][1] >= min_share
    if mixed:
        out += [b for b in families[ranked[1][0]][1] if b not in out]
    return out, mixed
