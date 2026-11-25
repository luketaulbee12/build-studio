"""Partial-block shape matching: which Minecraft block shape best fits the model inside one block.

Each block is sampled on an N x N x N grid (N = 4). Every candidate shape is a template on the same grid;
the best template (overlap score, with a small preference for simple shapes) decides the block's shape.
Index order of a cell grid: [x, y, z]  (+x east, +y up, +z south).
"""
import numpy as np

N = 4
H = N // 2
SIDES = {"north": (2, 0), "south": (2, N - 1), "west": (0, 0), "east": (0, N - 1)}  # axis, index
OPP = {"north": "south", "south": "north", "east": "west", "west": "east"}


def _templates():
    t = {}
    full = np.ones((N, N, N), bool)
    t["full"] = full
    b = np.zeros_like(full); b[:, :H, :] = True
    t["slab_bottom"] = b
    t["slab_top"] = ~b
    for f, (ax, i) in SIDES.items():
        side = np.zeros_like(full)
        sl = [slice(None)] * 3
        sl[ax] = slice(0, H) if i == 0 else slice(H, N)
        side[tuple(sl)] = True
        # stairs: the tall back is on the "facing" side
        t[f"stairs_{f}_bottom"] = b | (side & ~b)
        t[f"stairs_{f}_top"] = ~b | (side & b)
        # thin plate against one side (trapdoor open / thin wall)
        plate = np.zeros_like(full)
        sl = [slice(None)] * 3
        sl[ax] = slice(i, i + 1)
        plate[tuple(sl)] = True
        t[f"plate_{f}"] = plate
    post = np.zeros_like(full); post[H - 1:H + 1, :, H - 1:H + 1] = True
    t["post"] = post
    floor = np.zeros_like(full); floor[:, 0, :] = True
    t["floor"] = floor
    pane_x = np.zeros_like(full); pane_x[:, :, H - 1:H + 1] = True  # thin wall running east-west
    pane_z = np.zeros_like(full); pane_z[H - 1:H + 1, :, :] = True
    t["pane_x"], t["pane_z"] = pane_x, pane_z
    return t


TEMPLATES = _templates()
NAMES = list(TEMPLATES)
STACK = np.stack([TEMPLATES[n] for n in NAMES]).reshape(len(NAMES), -1).astype(np.float32)
# small handicap for fussier shapes, so a nearly-full block stays a full block
PENALTY = np.array([0.0 if n == "full" else 0.04 if n.startswith("slab") else 0.06 if n.startswith("stairs")
                    else 0.10 for n in NAMES], np.float32)


def match(cells):
    """cells: (M, N, N, N) bool occupancy. Returns (shape name per cell, score)."""
    occ = cells.reshape(len(cells), -1).astype(np.float32)
    inter = occ @ STACK.T
    union = occ.sum(1, keepdims=True) + STACK.sum(1)[None] - inter
    iou = inter / np.maximum(union, 1e-6) - PENALTY[None]
    best = iou.argmax(1)
    return [NAMES[i] for i in best], iou[np.arange(len(best)), best]


# ------------------------------------------------------------------------------- block ids per material

WOOD = {"oak": "oak", "spruce": "spruce", "birch": "birch", "dark_oak": "dark_oak", "jungle": "jungle",
        "acacia": "acacia", "mangrove": "mangrove", "cherry": "cherry"}
WALLS = {"cobblestone": "cobblestone_wall", "mossy_cobblestone": "mossy_cobblestone_wall",
         "stone_bricks": "stone_brick_wall", "mossy_stone_bricks": "mossy_stone_brick_wall",
         "cracked_stone_bricks": "stone_brick_wall", "andesite": "andesite_wall", "polished_andesite": "andesite_wall",
         "diorite": "diorite_wall", "granite": "granite_wall", "tuff": "tuff_wall", "tuff_bricks": "tuff_brick_wall",
         "deepslate_bricks": "deepslate_brick_wall", "deepslate_tiles": "deepslate_tile_wall",
         "cobbled_deepslate": "cobbled_deepslate_wall", "polished_deepslate": "polished_deepslate_wall",
         "bricks": "brick_wall", "mud_bricks": "mud_brick_wall", "sandstone": "sandstone_wall",
         "stone": "cobblestone_wall", "smooth_stone": "stone_brick_wall", "calcite": "diorite_wall",
         "smooth_sandstone": "sandstone_wall", "cut_sandstone": "sandstone_wall", "red_sandstone": "red_sandstone_wall",
         "cut_red_sandstone": "red_sandstone_wall", "end_stone_bricks": "end_stone_brick_wall",
         "polished_granite": "granite_wall", "polished_diorite": "diorite_wall", "blackstone": "blackstone_wall",
         "polished_blackstone_bricks": "polished_blackstone_brick_wall", "nether_bricks": "nether_brick_wall",
         "prismarine_bricks": "prismarine_wall"}


def wood_of(name):
    if "planks" not in name and "log" not in name:
        return None
    base = name.replace("stripped_", "").replace("_planks", "").replace("_log", "")
    return WOOD.get(base)


def block_for(shape, name, blocks):
    """Minecraft block state (without connection properties) for a shape made of material `name`.
    blocks: the converter's BLOCKS table (name -> (texture, stairs, slab)). Returns (state, family)."""
    _, stairs, slab = blocks.get(name, (None, None, None))
    wood = wood_of(name)
    if wood:
        stairs, slab = stairs or f"{wood}_stairs", slab or f"{wood}_slab"
    if shape == "full":
        s = "minecraft:" + name
        if "leaves" in name:
            s += "[persistent=true]"
        elif "log" in name:
            s += "[axis=y]"
        return s, None
    if "leaves" in name:  # foliage keeps its leafy look whatever the shape
        return ("minecraft:moss_carpet" if shape == "floor" else f"minecraft:{name}[persistent=true]"), None
    if shape.startswith("slab"):
        if slab:
            return f"minecraft:{slab}[type={shape[5:]}]", None
        return None, None
    if shape.startswith("stairs"):
        _, facing, half = shape.split("_")
        if stairs:
            return f"minecraft:{stairs}[facing={facing},half={half}]", None
        if slab:
            return f"minecraft:{slab}[type={half}]", None
        return None, None
    if shape == "post" or shape.startswith("pane"):
        if wood:
            return f"minecraft:{wood}_fence", "fence"
        if name in WALLS:
            return f"minecraft:{WALLS[name]}", "wall"
        return None, None
    if shape.startswith("plate"):
        side = shape.split("_")[1]
        if wood:  # open trapdoor lies against the side opposite its facing
            return f"minecraft:{wood}_trapdoor[facing={OPP[side]},half=bottom,open=true]", None
        return None, None
    if shape == "floor":
        if wood:
            return f"minecraft:{wood}_trapdoor[facing=north,half=bottom,open=false]", None
        if slab:
            return f"minecraft:{slab}[type=bottom]", None
    return None, None


def connect(states, family, solid_full):
    """Add fence / wall / pane connections: connect to the same family or to full blocks."""
    dirs = {"north": (0, 0, -1), "south": (0, 0, 1), "west": (-1, 0, 0), "east": (1, 0, 0)}
    out = dict(states)
    for p, fam in family.items():
        if not fam:
            continue
        props = []
        for d, (dx, dy, dz) in dirs.items():
            q = (p[0] + dx, p[1] + dy, p[2] + dz)
            on = family.get(q) == fam or q in solid_full or (fam == "pane" and family.get(q) == "pane")
            if fam == "wall":
                props.append(f"{d}={'low' if on else 'none'}")
            else:
                props.append(f"{d}={'true' if on else 'false'}")
        if fam == "wall":
            props.append("up=true")
        out[p] = states[p] + "[" + ",".join(sorted(props)) + "]"
    return out
