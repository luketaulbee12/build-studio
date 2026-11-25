"""Minecraft 1.12 numeric block ids (id, meta) -> modern block states, for old datasets like CraftAssist.
Covers the blocks builders actually use; anything else returns None (dropped)."""
COLORS = ["white", "orange", "magenta", "light_blue", "yellow", "lime", "pink", "gray", "light_gray", "cyan",
          "purple", "blue", "brown", "green", "red", "black"]
WOODS = ["oak", "spruce", "birch", "jungle", "acacia", "dark_oak"]
STAIR_FACING = ["east", "west", "south", "north"]

SIMPLE = {
    2: "grass_block", 4: "cobblestone", 7: "bedrock", 13: "gravel", 14: "gold_ore", 15: "iron_ore", 16: "coal_ore",
    20: "glass", 22: "lapis_block", 41: "gold_block", 42: "iron_block", 45: "bricks", 47: "bookshelf",
    48: "mossy_cobblestone", 49: "obsidian", 50: "torch", 54: "chest", 57: "diamond_block", 58: "crafting_table",
    61: "furnace", 65: "ladder", 78: "snow", 79: "ice", 80: "snow_block", 81: "cactus", 82: "clay", 85: "oak_fence",
    86: "carved_pumpkin", 87: "netherrack", 88: "soul_sand", 89: "glowstone", 91: "jack_o_lantern", 101: "iron_bars",
    102: "glass_pane", 103: "melon", 106: "vine", 107: "oak_fence_gate", 110: "mycelium", 111: "lily_pad",
    112: "nether_bricks", 113: "nether_brick_fence", 121: "end_stone", 133: "emerald_block", 152: "redstone_block",
    169: "sea_lantern", 170: "hay_block", 172: "terracotta", 173: "coal_block", 174: "packed_ice",
    188: "spruce_fence", 189: "birch_fence", 190: "jungle_fence", 191: "dark_oak_fence", 192: "acacia_fence",
    201: "purpur_block", 206: "end_stone_bricks", 214: "nether_wart_block", 215: "red_nether_bricks",
    31: "short_grass", 37: "dandelion", 38: "poppy", 30: "cobweb", 46: "tnt", 116: "enchanting_table",
    118: "cauldron", 145: "anvil", 25: "note_block", 84: "jukebox", 123: "redstone_lamp", 165: "slime_block",
    9: "water", 8: "water", 11: "lava", 10: "lava", 3: "dirt", 12: "sand", 24: "sandstone", 179: "red_sandstone",
}
STAIRS = {53: "oak_stairs", 67: "cobblestone_stairs", 108: "brick_stairs", 109: "stone_brick_stairs",
          114: "nether_brick_stairs", 128: "sandstone_stairs", 134: "spruce_stairs", 135: "birch_stairs",
          136: "jungle_stairs", 156: "quartz_stairs", 163: "acacia_stairs", 164: "dark_oak_stairs",
          180: "red_sandstone_stairs", 203: "purpur_stairs"}
STONE = ["stone", "granite", "polished_granite", "diorite", "polished_diorite", "andesite", "polished_andesite"]
STONE_SLABS = ["smooth_stone_slab", "sandstone_slab", "oak_slab", "cobblestone_slab", "brick_slab",
               "stone_brick_slab", "nether_brick_slab", "quartz_slab"]
COLORED = {35: "{}_wool", 95: "{}_stained_glass", 159: "{}_terracotta", 160: "{}_stained_glass_pane",
           171: "{}_carpet", 251: "{}_concrete", 252: "{}_concrete_powder"}


def to_state(bid, meta):
    bid, meta = int(bid), int(meta)
    if bid == 0:
        return None
    if bid in COLORED:
        return "minecraft:" + COLORED[bid].format(COLORS[meta & 15])
    if bid in STAIRS:
        return (f"minecraft:{STAIRS[bid]}[facing={STAIR_FACING[meta & 3]},half={'top' if meta & 4 else 'bottom'},"
                "shape=straight,waterlogged=false]")
    if bid == 1:
        return "minecraft:" + STONE[meta % 7]
    if bid == 5:
        return f"minecraft:{WOODS[meta % 6]}_planks"
    if bid in (17, 162):
        wood = WOODS[(meta & 3) + (4 if bid == 162 else 0)] if (meta & 3) + (4 if bid == 162 else 0) < 6 else "oak"
        axis = ["y", "x", "z", "y"][(meta >> 2) & 3]
        return f"minecraft:{wood}_log[axis={axis}]"
    if bid in (18, 161):
        wood = WOODS[(meta & 3) + (4 if bid == 161 else 0)] if (meta & 3) + (4 if bid == 161 else 0) < 6 else "oak"
        return f"minecraft:{wood}_leaves[persistent=true]"
    if bid == 98:
        return "minecraft:" + ["stone_bricks", "mossy_stone_bricks", "cracked_stone_bricks", "chiseled_stone_bricks"][meta & 3]
    if bid == 155:
        return "minecraft:" + ["quartz_block", "chiseled_quartz_block", "quartz_pillar[axis=y]", "quartz_pillar[axis=x]",
                               "quartz_pillar[axis=z]"][min(meta, 4)]
    if bid == 168:
        return "minecraft:" + ["prismarine", "prismarine_bricks", "dark_prismarine"][min(meta, 2)]
    if bid == 44:
        return f"minecraft:{STONE_SLABS[meta & 7]}[type={'top' if meta & 8 else 'bottom'}]"
    if bid == 126:
        return f"minecraft:{WOODS[(meta & 7) % 6]}_slab[type={'top' if meta & 8 else 'bottom'}]"
    if bid == 43:
        return "minecraft:" + ["smooth_stone", "sandstone", "oak_planks", "cobblestone", "bricks", "stone_bricks",
                               "nether_bricks", "quartz_block"][meta & 7]
    if bid == 125:
        return f"minecraft:{WOODS[(meta & 7) % 6]}_planks"
    if bid == 139:
        return "minecraft:" + ("mossy_cobblestone_wall" if meta & 1 else "cobblestone_wall")
    if bid == 96:
        return "minecraft:oak_trapdoor"
    if bid in (64, 193, 194, 195, 196, 197):
        wood = {64: "oak", 193: "spruce", 194: "birch", 195: "jungle", 196: "acacia", 197: "dark_oak"}[bid]
        return f"minecraft:{wood}_door[half={'upper' if meta & 8 else 'lower'}]"
    if bid in SIMPLE:
        return "minecraft:" + SIMPLE[bid]
    return None
