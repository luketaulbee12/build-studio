"""Grabcraft display names ("Spruce Wood Slab (Upper)") -> modern block states, checked against the game's
block list. Unknown names return None."""
import re
from pathlib import Path

_KNOWN = None
WOODS = ("oak", "spruce", "birch", "jungle", "acacia", "dark_oak", "mangrove", "cherry")
FIX = {"grass": "grass_block", "still_water": "water", "water": "water", "nether_brick": "nether_bricks",
       "double_stone_slab": "smooth_stone", "smooth_double_stone_slab": "smooth_stone", "stone_slab": "smooth_stone_slab",
       "hardened_clay": "terracotta", "clay": "clay", "brick": "bricks", "stone_brick": "stone_bricks",
       "end_stone_brick": "end_stone_bricks", "quartz_pillar_block": "quartz_pillar", "pillar_quartz_block": "quartz_pillar",
       "mossy_stone_brick": "mossy_stone_bricks", "cracked_stone_brick": "cracked_stone_bricks",
       "chiseled_stone_brick": "chiseled_stone_bricks", "wooden_plank": "oak_planks", "red_nether_brick": "red_nether_bricks",
       "lapis_lazuli_block": "lapis_block", "block_of_quartz": "quartz_block", "torch": "torch",
       "cobweb": "cobweb", "web": "cobweb", "carpet": "white_carpet", "wool": "white_wool", "planks": "oak_planks"}


def _known():
    global _KNOWN
    if _KNOWN is None:
        d = Path(__file__).resolve().parent.parent / "studio" / "data" / "mc" / "blockstates"
        _KNOWN = {f.stem for f in d.glob("*.json")}
    return _KNOWN


def to_state(display):
    s = display.strip()
    props = " ".join(re.findall(r"\(([^)]*)\)?", s)).lower()
    base = re.sub(r"\(.*", "", s).strip().lower()
    base = base.replace("wooden", "wood").replace(" ", "_")
    base = FIX.get(base, base)
    # "<wood>_wood_plank(s)", "<wood>_wood_slab", "<wood>_wood_stairs", "<wood>_wood" (log), "<wood>_wood_fence"
    for w in WOODS:
        if base.startswith(w + "_wood"):
            rest = base[len(w) + 5:].lstrip("_")
            base = {"plank": f"{w}_planks", "planks": f"{w}_planks", "slab": f"{w}_slab", "stairs": f"{w}_stairs",
                    "fence": f"{w}_fence", "fence_gate": f"{w}_fence_gate", "": f"{w}_log", "log": f"{w}_log",
                    "door": f"{w}_door", "trapdoor": f"{w}_trapdoor", "button": f"{w}_button"}.get(rest, base)
            break
    base = base.replace("stained_clay", "terracotta").replace("_brick_", "_bricks_") if "stained_clay" in base else base
    if base.endswith("_leaves"):
        return f"minecraft:{base}[persistent=true]" if base in _known() else None
    if base not in _known():
        if base.endswith("s") and base[:-1] in _known():
            base = base[:-1]
        elif base + "s" in _known():
            base += "s"
        else:
            return None
    st = "minecraft:" + base
    if base.endswith("_slab"):
        st += "[type=top]" if "upper" in props else "[type=bottom]"
    elif base.endswith("_stairs"):
        f = next((d for d in ("north", "south", "east", "west") if d in props), "north")
        st += f"[facing={f},half={'top' if 'upside' in props else 'bottom'}]"
    elif base.endswith("_log"):
        st += "[axis=z]" if "north/south" in props else "[axis=x]" if "east/west" in props else "[axis=y]"
    elif base.endswith("_door"):
        st += "[half=upper]" if "upper" in props else "[half=lower]"
    return st
