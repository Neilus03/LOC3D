"""Portable preparation and checkpoint admission for matched sampling studies."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from delimit3d.runtime.config import METADATA_KEYS, expand_environment, load_config
from delimit3d.runtime.records import file_sha256, write_json
from delimit3d.training.logical_batch_schedule import LogicalBatchSchedule
from delimit3d.training.sampling_ablation import VARIANTS, REFERENCE_FEATURE_HARD_ACTIVATION_UPDATE


def _split(path):
    scenes = [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]
    if not scenes or len(set(scenes)) != len(scenes):
        raise ValueError("Expected a nonempty split without duplicate scene IDs")
    if any(Path(scene).name != scene or scene in {".", ".."} for scene in scenes):
        raise ValueError("Invalid scene ID in split")
    return scenes


def _checked_artifact(manifest, record):
    path = Path(record["path"])
    path = (path if path.is_absolute() else manifest.parent / path).resolve(strict=True)
    if path.stat().st_size != record["size_bytes"] or file_sha256(path) != record["sha256"]:
        raise ValueError(f"Source artifact changed: {path}")
    return path


def audit_areas(source_root, train_split, source_audit):
    """Read original 2D areas from optimization scenes and training frames only."""
    import numpy as np
    source_root, train_split, source_audit = map(Path, (source_root, train_split, source_audit))
    audit = json.loads(source_audit.read_text())
    if audit.get("passed") is not True or audit.get("artifact_hashes_verified") is not True:
        raise ValueError("A passed, artifact-verified source audit is required")
    if audit["splits"]["optimization"]["sha256"] != file_sha256(train_split):
        raise ValueError("Training split differs from the source audit")
    sealed = {row["scene_id"]: row["manifest_sha256"] for row in audit["scenes"]}
    chunks, records, image_shapes = [], [], set()
    for scene in _split(train_split):
        path = source_root / scene / "source_manifest.json"
        if file_sha256(path) != sealed.get(scene):
            raise ValueError(f"Scene manifest differs from source audit: {scene}")
        manifest = json.loads(path.read_text())
        if manifest.get("scene_id") != scene or manifest.get("complete") is not True:
            raise ValueError(f"Invalid source manifest: {scene}")
        cache_path = _checked_artifact(path, manifest["artifacts"]["sam2_projection_cache"])
        annotation_path = _checked_artifact(path, manifest["artifacts"]["annotation_manifest"])
        annotations = json.loads(annotation_path.read_text())
        by_frame = {row["frame_id"]: row for row in annotations["frames"]}
        if len(by_frame) != len(annotations["frames"]):
            raise ValueError("Duplicate annotation frame IDs")
        with np.load(cache_path, allow_pickle=False) as data:
            ids = [str(value) for value in data["physical_frame_ids"]]
            splits = data["split_codes"]
            offsets, areas = data["proposal_frame_offsets"], data["proposal_areas"]
            trainable = data["proposal_trainable"]
            if (str(data["scene_id"].item()) != scene or len(set(ids)) != len(ids)
                    or splits.shape != (len(ids),) or not np.isin(splits, (0, 1)).all()
                    or offsets.shape != (len(ids) + 1,) or offsets.dtype.kind not in "iu"
                    or offsets[0] != 0 or offsets[-1] != len(areas) or np.any(np.diff(offsets) < 0)
                    or areas.ndim != 1 or areas.dtype.kind not in "iu"
                    or trainable.shape != areas.shape or trainable.dtype != np.bool_):
                raise ValueError(f"Invalid area cache: {scene}")
            selected, frame_count = [], 0
            for index, frame in enumerate(ids):
                row = by_frame[frame]
                if (row["split"] == "train") != (int(splits[index]) == 0):
                    raise ValueError("Annotation/cache frame split mismatch")
                if int(splits[index]) != 0:
                    continue
                shape = tuple(row["image_shape"])
                if len(shape) != 2 or any(type(v) is not int or v <= 0 for v in shape):
                    raise ValueError("Invalid image shape")
                image_shapes.add(shape)
                lo, hi = offsets[index:index + 2]
                values = areas[lo:hi][trainable[lo:hi]]
                if np.any((values <= 0) | (values > shape[0] * shape[1])):
                    raise ValueError("Invalid original 2D proposal area")
                selected.append(values)
                frame_count += 1
            values = np.concatenate(selected) if selected else np.empty(0, dtype=np.int64)
            chunks.append(values)
            records.append({"scene": scene, "train_frames": frame_count,
                            "trainable_proposals": int(len(values)),
                            "projection_cache_sha256": file_sha256(cache_path),
                            "annotation_manifest_sha256": file_sha256(annotation_path)})
    if len(image_shapes) != 1:
        raise ValueError("Pixel-area quartiles require a common image resolution")
    values = np.concatenate(chunks)
    if not len(values):
        raise ValueError("No trainable proposals")
    cutoff = int(np.quantile(values, .25, method="lower"))
    removed = int(np.count_nonzero(values <= cutoff))
    return {"schema": "loc3d_train_proposal_area_audit/v1", "passed": True,
            "image_shape": list(next(iter(image_shapes))), "threshold_pixels_inclusive": cutoff,
            "definition": "global lower quartile of train-frame projected proposals marked trainable; remove area <= threshold from positive sampling only",
            "ties_policy": "remove all proposals tied at cutoff; report actual fraction",
            "proposals": int(len(values)), "removed": removed, "removed_fraction": removed / len(values),
            "train_scenes": len(records), "train_frames": sum(row["train_frames"] for row in records),
            "train_split_sha256": file_sha256(train_split), "source_audit_sha256": file_sha256(source_audit),
            "scenes": records}


def prepare_study(config, output, initial_checkpoint, area_audit, *, stop_epoch=50,
                  activation_update=REFERENCE_FEATURE_HARD_ACTIVATION_UPDATE):
    """Write five matched recipes; retain the reference's nominal LR horizon."""
    output = Path(output).resolve()
    values = load_config(config, expand_env=False)
    values.update(output_dir=str(output / "runs/full"), initial_checkpoint=str(Path(initial_checkpoint).resolve(strict=True)))
    values = expand_environment(values)
    required = {"source_granularity": "sam2", "allow_shared_proposal_negatives": True,
                "sam2_maximum_raw_points": 100000, "epochs": 150, "expected_world_size": 2,
                "scenes_per_rank_per_update": 6, "sampling_profile": "pf_hnm_512_p16_v1",
                "flush_logical_batch_at_eval": True}
    if any(values.get(key) != value for key, value in required.items()):
        raise ValueError("Use the selected-proposal 100k/150-epoch two-rank SAM2 reference recipe")
    scenes = _split(values["train_split"])
    heldout = _split(values["holdout_split"])
    if len(scenes) < values["expected_world_size"]:
        raise ValueError("Each training rank requires at least one canonical scene")
    if set(scenes) & set(heldout):
        raise ValueError("Optimization and validation scenes overlap")
    flush = tuple(int(value) for value in str(values["evaluation_epochs"]).split(",") if int(value) > 0)
    schedule = LogicalBatchSchedule((len(scenes) + 1) // 2, 6, 150, flush)
    if stop_epoch not in flush:
        raise ValueError("Stop epoch must already be a reference evaluation boundary; do not change flushes")
    if type(activation_update) is not int or not 0 < activation_update <= schedule.total_updates:
        raise ValueError("Invalid reference feature-hard activation update")
    area_audit = Path(area_audit).resolve(strict=True)
    area = json.loads(area_audit.read_text())
    bindings = {key: {"path": str(Path(values[key]).resolve(strict=True)), "sha256": file_sha256(values[key])}
                for key in ("train_split", "holdout_split", "source_audit", "code_manifest", "initial_checkpoint")}
    fixed = Path(values["fixed_eval_pack"]) / "manifest.json"
    bindings["fixed_eval_manifest"] = {"path": str(fixed.resolve(strict=True)), "sha256": file_sha256(fixed)}
    bindings["proposal_area_audit"] = {"path": str(area_audit), "sha256": file_sha256(area_audit)}
    # Validate all recipes before publishing the study index.
    output.mkdir(parents=True, exist_ok=False)
    from delimit3d.pretraining.arguments import parse_args
    recipes = {}
    for variant in VARIANTS:
        recipe = {key: value for key, value in values.items() if key not in METADATA_KEYS}
        recipe.update(ablation_variant=variant, ablation_activation_update=activation_update,
                      pause_after_update=schedule.update_at_epoch(stop_epoch),
                      output_dir=str(output / "runs" / variant))
        if variant == "no_small_positive":
            recipe.update(proposal_area_cutoff=area["threshold_pixels_inclusive"], proposal_area_audit=str(area_audit))
        path = output / "configs" / f"{variant}.json"
        write_json(path, recipe)
        parse_args(["--config", str(path)])
        recipes[variant] = {"path": str(path), "sha256": file_sha256(path)}
    result = {"schema": "loc3d_sampling_study/v1", "status": "PREPARED", "recipes": recipes,
              "inputs": bindings, "stop_epoch": stop_epoch, "stop_update": schedule.update_at_epoch(stop_epoch),
              "logical_batch_schedule": schedule.as_dict(), "feature_hard_activation_update": activation_update,
              "reference_gate": "adaptive; verify its observed activation matches this controlled schedule",
              "claim_boundary": "One pretraining seed. Preparation is not GPU execution or evaluation evidence."}
    write_json(output / "study.json", result)
    return result


MATCHED_KEYS = ("logical_batch_schedule", "rank_balance_policy", "source_loading_mode", "world_size", "seed",
                "input_features", "grid_size", "epochs", "total_updates", "backbone_lr", "head_lr", "weight_decay",
                "warmup_updates", "scenes_per_rank_per_update", "normalization", "initialization_seed", "fixed_eval_seed",
                "source_granularity", "sampling_profile", "evaluation_sampling_profile", "train_augmentation_profile")


def _verify_recipe_settings(cfg, args, schedule):
    direct = ("rank_balance_policy", "source_loading_mode", "seed", "input_features", "grid_size", "epochs",
              "backbone_lr", "head_lr", "weight_decay", "scenes_per_rank_per_update", "source_granularity",
              "sampling_profile", "train_augmentation_profile", "dataset", "litept_variant", "sampling_mode",
              "coordinate_normalization_policy", "multiscale_supervision", "warmup_epochs")
    expected = {key: getattr(args, key) for key in direct}
    warmup = (args.warmup_epochs * schedule.scenes_per_epoch + schedule.nominal_batch_size - 1) // schedule.nominal_batch_size
    expected.update(world_size=args.expected_world_size, total_updates=schedule.total_updates,
                    warmup_updates=warmup, precision="fp32", token_pure_dec0=True, supervised_stages=["dec0"],
                    evaluation_sampling_profile=args.evaluation_sampling_profile or args.sampling_profile,
                    initialization_seed=args.seed if args.initialization_seed is None else args.initialization_seed,
                    fixed_eval_seed=args.seed if args.fixed_eval_seed is None else args.fixed_eval_seed,
                    rgb_color_mean=args.color_mean, rgb_color_std=args.color_std,
                    eval_epochs=list(args.evaluation_epochs), checkpoint_epochs=list(args.checkpoint_epochs),
                    scene_visits_per_epoch_per_rank=schedule.scenes_per_epoch)
    if changed := [key for key, value in expected.items() if cfg.get(key) != value]:
        raise ValueError(f"Checkpoint settings differ from prepared recipe: {changed}")
    normalization = cfg.get("normalization", {})
    if (normalization.get("policy") != args.normalization_policy
            or normalization.get("bn_recalibration_scenes") != args.bn_recalibration_scenes
            or cfg.get("hierarchy_supervision", {}).get("mode") != "off"):
        raise ValueError("Checkpoint normalization or objective differs from prepared recipe")
    lr = cfg.get("lr_schedule", {})
    expected_lr = {"kind": "linear_warmup_cosine", "start_epoch": 0, "start_update": 0,
                   "warmup_epochs": args.warmup_epochs, "warmup_updates": warmup,
                   "start_factor": 0.0, "peak_factor": 1.0, "final_factor": .01}
    if lr != expected_lr:
        raise ValueError("Checkpoint LR horizon differs from prepared recipe")


def verify_checkpoints(study_path, *, epoch, update, checkpoints=None, interim=False):
    """Admit exact finite checkpoints; never label this feature evaluation completion."""
    import torch
    from delimit3d.data.sam2_source import recorded_point_limit
    study_path = Path(study_path).resolve(strict=True)
    study = json.loads(study_path.read_text())
    if study.get("schema") != "loc3d_sampling_study/v1" or study.get("status") != "PREPARED":
        raise ValueError("Expected a prepared sampling study")
    if type(epoch) is not int or type(update) is not int or epoch <= 0 or update <= 0:
        raise ValueError("Checkpoint epoch/update must be positive integers")
    if (epoch, update) != (study["stop_epoch"], study["stop_update"]) and not interim:
        raise ValueError("Non-endpoint checkpoints must be explicitly marked interim")
    schedule = LogicalBatchSchedule.from_dict(study["logical_batch_schedule"])
    if epoch > study["stop_epoch"] or schedule.update_at_epoch(epoch) != update:
        raise ValueError("Checkpoint epoch/update does not match the prepared schedule")
    for record in (*study["inputs"].values(), *study["recipes"].values()):
        if file_sha256(record["path"]) != record["sha256"]:
            raise ValueError(f"Study input changed: {record['path']}")
    overrides = checkpoints or {}
    if set(overrides) - set(VARIANTS):
        raise ValueError("Unknown checkpoint variant")
    rows, common = {}, None
    for variant in VARIANTS:
        path = Path(overrides.get(variant, study_path.parent / "runs" / variant / "checkpoints" / f"epoch_{epoch:04d}.pt"))
        before = file_sha256(path)
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if file_sha256(path) != before:
            raise ValueError(f"Checkpoint changed while reading: {variant}")
        if (payload.get("epoch"), payload.get("update"), payload.get("resumable")) != (epoch, update, True):
            raise ValueError(f"Wrong checkpoint boundary: {variant}")
        cfg = payload["resolved_config"]
        from delimit3d.pretraining.arguments import parse_args
        args = parse_args(["--config", study["recipes"][variant]["path"]])
        _verify_recipe_settings(cfg, args, schedule)
        if cfg.get("ablation_variant", "full") != variant:
            raise ValueError(f"Wrong checkpoint sampling arm: {variant}")
        weights = payload.get("model_state_dict", {})
        if not weights or not all(isinstance(t, torch.Tensor) and torch.isfinite(t).all().item() for t in weights.values()):
            raise ValueError(f"Nonfinite or empty model state: {variant}")
        current = {key: cfg.get(key) for key in MATCHED_KEYS}
        if any(value is None for value in current.values()):
            raise ValueError(f"Checkpoint lacks matched protocol fields: {variant}")
        if current["logical_batch_schedule"] != study["logical_batch_schedule"]:
            raise ValueError(f"Checkpoint schedule differs from prepared study: {variant}")
        if recorded_point_limit(cfg) != 100000 or cfg.get("allow_shared_proposal_negatives") is not True:
            raise ValueError(f"Checkpoint cap/fixed-evaluation policy differs: {variant}")
        if cfg.get("initialization", {}).get("sha256") != study["inputs"]["initial_checkpoint"]["sha256"]:
            raise ValueError(f"Checkpoint initialization differs: {variant}")
        if cfg.get("fixed_eval_pack", {}).get("manifest_sha256") != study["inputs"]["fixed_eval_manifest"]["sha256"]:
            raise ValueError(f"Checkpoint fixed-evaluation pack differs: {variant}")
        for resolved_key, input_key in (("train_split", "train_split"), ("holdout_split", "holdout_split"),
                                        ("source_dataset_audit", "source_audit"), ("code_manifest", "code_manifest")):
            if cfg.get(resolved_key, {}).get("sha256") != study["inputs"][input_key]["sha256"]:
                raise ValueError(f"Checkpoint {resolved_key} differs from prepared inputs: {variant}")
        if common is None:
            common = current
        elif current != common:
            raise ValueError(f"Unmatched checkpoint training contract: {variant}")
        activation = study["feature_hard_activation_update"]
        gate = payload.get("feature_hard_state", {})
        if update >= activation and (gate.get("enabled") is not True or gate.get("activation_update") != activation):
            raise ValueError(f"Reference feature-hard activation not matched: {variant}")
        if update < activation and (gate.get("enabled") is not False or gate.get("activation_update") is not None):
            raise ValueError(f"Unexpected early feature-hard activation: {variant}")
        if variant != "full":
            from delimit3d.training.sampling_ablation import ablation_contract
            expected = ablation_contract(args)
            if any(cfg.get(key) != value for key, value in expected.items()):
                raise ValueError(f"Checkpoint ablation policy differs: {variant}")
        rows[variant] = {"checkpoint": str(path.resolve()), "sha256": before, "epoch": epoch, "update": update}
    return {"schema": "loc3d_sampling_checkpoint_admission/v1", "status": "CHECKPOINTS_VERIFIED",
            "study_sha256": file_sha256(study_path), "interim": bool(interim), "checkpoints": rows,
            "feature_hard_has_activated": update >= study["feature_hard_activation_update"],
            "claim_boundary": "Checkpoint admission only; no frozen-feature metrics or evaluation completion established."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    area = commands.add_parser("area-audit", help="Seal the train-only lower-quartile 2D mask-area cutoff")
    for flag in ("source-root", "train-split", "source-audit", "output"):
        area.add_argument("--" + flag, type=Path, required=True)
    prep = commands.add_parser("prepare", help="Write five matched recipes without launching training")
    for flag in ("config", "initial-checkpoint", "area-audit", "output"):
        prep.add_argument("--" + flag, type=Path, required=True)
    prep.add_argument("--stop-epoch", type=int, default=50)
    prep.add_argument("--activation-update", type=int, default=REFERENCE_FEATURE_HARD_ACTIVATION_UPDATE)
    verify = commands.add_parser("verify-checkpoints", help="Verify all five checkpoints before separate evaluation")
    verify.add_argument("--study", type=Path, required=True)
    verify.add_argument("--epoch", type=int, required=True)
    verify.add_argument("--update", type=int, required=True)
    verify.add_argument("--checkpoint", action="append", default=[], metavar="VARIANT=PATH")
    verify.add_argument("--interim", action="store_true")
    verify.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.action == "area-audit":
        result = audit_areas(args.source_root, args.train_split, args.source_audit)
    elif args.action == "prepare":
        result = prepare_study(args.config, args.output, args.initial_checkpoint, args.area_audit,
                               stop_epoch=args.stop_epoch, activation_update=args.activation_update)
    else:
        overrides = {}
        for entry in args.checkpoint:
            variant, separator, path = entry.partition("=")
            if not separator or not path or variant in overrides:
                parser.error("Use unique --checkpoint VARIANT=PATH entries")
            overrides[variant] = path
        result = verify_checkpoints(args.study, epoch=args.epoch, update=args.update,
                                    checkpoints=overrides, interim=args.interim)
    if args.action != "prepare":
        write_json(args.output, result)
    print(json.dumps({key: value for key, value in result.items() if key not in {"scenes", "recipes", "inputs"}}, indent=2))


if __name__ == "__main__":
    main()
