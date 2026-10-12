"""Sonata Small transfers with the matched Concerto native-stage interface."""
from __future__ import annotations

from delimit3d.training.concerto_contract import (
    DIRECT_INPUT_PROJECTION, FEATURE_CHANNELS, LEGACY_INPUT_PROJECTION,
    NATIVE_STAGE_RECIPE, THREE_STAGE_RECIPE, concerto_feature_channels,
)

SONATA_SOURCE_COMMIT = "18c09ff8d713494f78a8213792262b910977a65d"
SONATA_CHECKPOINT_REVISION = "df99897472c09f91ba9288da0a034aacffc0b010"
SONATA_CHECKPOINT_SHA256 = "ee5dae277dde1d164499579a6b17d0bd38df4e4286b0f1ba21a82f8cc14270d1"
SONATA_ENCODER_CHANNELS = (32, 64, 128, 256, 512)
SONATA_ENCODER_DEPTHS = (2, 2, 2, 6, 2)
SONATA_ENCODER_PARAMETERS = 38_648_992


def validate_checkpoint_config(config):
    expected = {
        "in_channels": 9, "enc_mode": True, "traceable": True,
        "enc_channels": SONATA_ENCODER_CHANNELS,
        "enc_depths": SONATA_ENCODER_DEPTHS,
        "stride": (2, 2, 2, 2),
    }
    for key, value in expected.items():
        actual = config.get(key)
        if isinstance(value, tuple) and isinstance(actual, (list, tuple)):
            actual = tuple(actual)
        if actual != value:
            raise ValueError(f"Sonata Small checkpoint interface mismatch: {key}")


def sonata_feature_channels(recipe):
    if recipe != NATIVE_STAGE_RECIPE:
        raise ValueError("Sonata transfers require all-five finest features and native memory stages")
    return concerto_feature_channels(recipe)


def sonata_feature_recipe(config):
    recipe = config["sonata"]["feature_recipe"]
    sonata_feature_channels(recipe)
    return recipe


def sonata_input_projection(config):
    sonata_feature_recipe(config)
    if config["sonata"]["input_projection"] != DIRECT_INPUT_PROJECTION:
        raise ValueError("Sonata transfers require direct projections to the shared task head")
    return DIRECT_INPUT_PROJECTION


def sonata_voxel_size(config, family):
    observed = (config["protocol"]["voxel_size_m"] if family == "agile3d" else
                config["model"]["voxel_size"] if family == "mask3d" else None)
    if observed != .05 or config["sonata"].get("voxel_size_m", .05) != .05:
        raise ValueError("Matched Sonata transfers require the existing 5 cm downstream grid")
    return .05
