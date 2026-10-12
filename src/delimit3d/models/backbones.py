"""Explicit backbone selection shared by task models and feature probes."""
from __future__ import annotations

from delimit3d.training.concerto_contract import (
    DIRECT_INPUT_PROJECTION, concerto_feature_channels,
)


def backbone_name(config):
    model = config.get("model", {})
    name = model.get("backbone_name", model.get("litept_variant", "litept_s_star"))
    if name == "concerto_small":
        return "concerto"
    if name == "sonata_small":
        return "sonata"
    if name not in {"litept_s_star", "concerto", "sonata"}:
        raise ValueError(f"Unsupported backbone: {name}")
    return name


def ptv3_recipe(config):
    name = backbone_name(config)
    return config[name].get("feature_recipe", "native_encoder_memory_with_all_five_finest_concat")


def ptv3_direct_inputs(config):
    return config[backbone_name(config)].get("input_projection", DIRECT_INPUT_PROJECTION) == DIRECT_INPUT_PROJECTION


def build_ptv3_backbone(config, *, frozen, grid_size):
    name = backbone_name(config)
    settings = config[name]
    if name == "sonata":
        from delimit3d.models.sonata_wrapper import SonataBackbone
        constructor = SonataBackbone
    elif name == "concerto":
        from delimit3d.models.concerto_wrapper import ConcertoBackbone
        constructor = ConcertoBackbone
    else:
        raise ValueError("PTv3 factory requires Concerto or Sonata")
    return constructor(
        source_root=settings["source_root"], checkpoint=settings["checkpoint"],
        frozen=frozen, grid_size=grid_size, feature_recipe=ptv3_recipe(config),
        activation_checkpointing=settings.get("activation_checkpointing", False),
    )
