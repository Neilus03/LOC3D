"""Explicit feature and voxel contracts for Concerto downstream experiments."""
from __future__ import annotations


THREE_STAGE_RECIPE = "official_three_deepest_stages_concat_then_exact_unpool"
ALL_FIVE_STAGE_RECIPE = "official_all_five_stages_concat_then_exact_unpool"
NATIVE_STAGE_RECIPE = "native_encoder_memory_with_all_five_finest_concat"
LEGACY_INPUT_PROJECTION = "litept_sstar_compatibility"
DIRECT_INPUT_PROJECTION = "native_to_hidden"
FEATURE_CHANNELS = {
    THREE_STAGE_RECIPE: (896, 896, 896, 768, 512),
    ALL_FIVE_STAGE_RECIPE: (992, 960, 896, 768, 512),
    NATIVE_STAGE_RECIPE: (992, 64, 128, 256, 512),
}


def concerto_feature_channels(recipe):
    """Return widths in finest-to-coarsest order without changing grid spacing."""
    if recipe not in FEATURE_CHANNELS:
        raise ValueError(f"Unknown Concerto feature recipe: {recipe!r}")
    return FEATURE_CHANNELS[recipe]


def concerto_feature_recipe(config):
    """Historical configurations retain their original three-stage features."""
    recipe = config["concerto"].get("feature_recipe", THREE_STAGE_RECIPE)
    concerto_feature_channels(recipe)
    return recipe


def concerto_input_projection(config):
    """Native features opt into direct task-head projections explicitly."""
    mode = config["concerto"].get("input_projection", LEGACY_INPUT_PROJECTION)
    if mode not in (LEGACY_INPUT_PROJECTION, DIRECT_INPUT_PROJECTION):
        raise ValueError(f"Unknown Concerto input projection: {mode!r}")
    recipe = concerto_feature_recipe(config)
    if recipe == NATIVE_STAGE_RECIPE and mode != DIRECT_INPUT_PROJECTION:
        raise ValueError("Native encoder stages require direct task-head projections")
    if mode == DIRECT_INPUT_PROJECTION and recipe == THREE_STAGE_RECIPE:
        raise ValueError("Direct task-head projections require all five encoder stages")
    return mode


def concerto_voxel_size(config, family):
    """Default historical runs to 5 cm; require explicit opt-in to 2 cm."""
    expected = float(config["concerto"].get("voxel_size_m", .05))
    if expected not in (.02, .05):
        raise ValueError("Concerto comparisons support explicit 2 cm or 5 cm grids")
    if family == "agile3d":
        observed = float(config["protocol"]["voxel_size_m"])
    elif family == "mask3d":
        observed = float(config["model"]["voxel_size"])
    else:
        raise ValueError(f"Unknown Concerto task: {family}")
    if observed != expected:
        raise ValueError("Downstream and Concerto encoder voxel sizes disagree")
    if expected == .02 and not config["concerto"].get("resolution_control"):
        raise ValueError("A 2 cm run requires a recorded resolution-control contract")
    return expected
