"""YAML/JSON defaults for the pretraining CLI; explicit flags take precedence."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Sequence

from delimit3d.runtime.config import METADATA_KEYS, expand_environment, load_config


def parse_with_config(
    parser: argparse.ArgumentParser, argv: Sequence[str] | None = None
) -> argparse.Namespace:
    probe = argparse.ArgumentParser(add_help=False)
    probe.add_argument("--config", type=Path)
    config_path = probe.parse_known_args(argv)[0].config
    if config_path is not None:
        try:
            values = load_config(config_path, expand_env=False)
            actions = {action.dest: action for action in parser._actions}
            supplied_options = {
                token.split("=", 1)[0]
                for token in (sys.argv[1:] if argv is None else argv)
                if token.startswith("-")
            }
            defaults = {}
            for key, value in values.items():
                if key in METADATA_KEYS:
                    continue
                if key not in actions or key in {"help", "config"}:
                    raise ValueError(f"unknown configuration key: {key}")
                action = actions[key]
                if supplied_options.intersection(action.option_strings):
                    continue
                value = expand_environment(value)
                if value is None and action.default is None and not action.required:
                    defaults[key] = None
                    continue
                if key in {"hierarchy_mask_stage_weights", "hierarchy_mask_stage_weight_schedule"} and not isinstance(value, str):
                    value = json.dumps(value, sort_keys=True)
                if action.nargs == 0:
                    if not isinstance(value, bool):
                        raise ValueError(f"{key} must be a boolean")
                    converted = value
                else:
                    is_list = action.nargs is not None
                    items = value if is_list else [value]
                    if not isinstance(items, list) or not items:
                        raise ValueError(f"{key} must be a nonempty list")
                    if isinstance(action.nargs, int) and len(items) != action.nargs:
                        raise ValueError(f"{key} requires {action.nargs} values")
                    items = [action.type(item) if action.type else item for item in items]
                    if action.choices is not None and any(
                        item not in action.choices for item in items
                    ):
                        raise ValueError(f"{key} must be one of {action.choices}")
                    converted = items if is_list else items[0]
                defaults[key] = converted
                action.required = False
            parser.set_defaults(**defaults)
        except (OSError, ValueError, TypeError) as error:
            parser.error(f"{config_path}: {error}")
    args = parser.parse_args(argv)
    if getattr(args, "allow_shared_proposal_negatives", False) and (
        args.source_granularity != "sam2" or args.hierarchy_supervision != "off"
        or args.multiscale_supervision or args.sampling_mode != "coverage_multigranular_v2"
    ):
        parser.error("Shared-proposal negatives are restricted to the SAM2 dec0-only ablation")
    if getattr(args, "allow_shared_proposal_negatives", False) and args.fixed_eval_pack is None:
        parser.error("Selected-proposal negatives require a matching fixed-evaluation pack")
    if not math.isfinite(args.grid_size) or args.grid_size <= 0:
        parser.error("--grid-size must be finite and positive")
    # Check before DDP initialization so invalid presets fail without GPUs.
    if getattr(args, "flush_logical_batch_at_eval", False) and (
        args.source_granularity != "sam2" or args.maximum_updates is not None
        or args.extension_start_epoch is not None
    ):
        parser.error("Checkpoint-boundary batch flushing requires SAM2 without funnel/extension mode")
    if getattr(args, "source_granularity", None) is not None:
        if args.source_family != "2d" or args.sampling_mode != "coverage_multigranular_v2":
            parser.error("--source-granularity requires source family 2d and V2 sampling")
        if args.hierarchy_supervision != "off" or args.multiscale_supervision:
            parser.error("--source-granularity requires dec0-only supervision")
        if args.source_granularity == "sam2" and (
            args.dataset != "structured3d" or args.litept_variant not in ("litept_b_star", "litept_s_star")
            or args.input_features != "rgbn6" or args.required_crop_mode != "full_scene"
            or args.train_augmentation_profile != "mask3d_scannet_rgbn6_fullscene_v1"
        ):
            parser.error("SAM2 preset requires Structured3D, LitePT-S*/B*, RGBN6 and full-scene geometry/augmentation")
    if getattr(args, "rank_balance_policy", "strict_even") != "strict_even" and (
        args.dataset != "structured3d" or args.source_granularity != "sam2"
        or args.litept_variant != "litept_s_star" or args.expected_world_size not in (2, 3, 4)
        or args.scenes_per_rank_per_update * args.expected_world_size != 12
        or not args.flush_logical_batch_at_eval
    ):
        parser.error("Rotating rank padding requires two/three/four-rank SAM2 LitePT-S* with global batch 12")
    return args
