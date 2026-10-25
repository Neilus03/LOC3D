"""Protocol validation and per-epoch observed-supervision audit for ScanNet LA."""
from __future__ import annotations

import hashlib


def enabled(config):
    return config.get("data", {}).get("limited_annotation") is not None


def validate_config(config):
    if not enabled(config):
        return
    data, model, train = config["data"], config["model"], config["train"]
    annotation = data["limited_annotation"]
    if not isinstance(annotation, dict) or annotation.get("budget_points") not in (20, 50, 100, 200):
        raise ValueError("Select official ScanNet LA points20/50/100/200")
    if not annotation.get("prepared_root"):
        raise ValueError("LA requires a sanitized prepared_root")
    checks = {
        "data": {"max_points": None, "sphere_point_max": None, "subsampling_mode": "none",
                 "use_colors": True, "use_normals": True, "append_xyz_to_features": False,
                 "augmentations": True, "augmentation_profile": "mask3d_scannet",
                 "color_mean": [0., 0., 0.], "color_std": [1., 1., 1.]},
        "model": {"voxel_size": .05, "voxel_reduce": "representative", "representative_sampling": "first",
                  "input_features": "rgbn6", "input_channels": 6, "mask_only": False,
                  "num_classes": 2, "train_queries": 100, "eval_queries": 100,
                  "feature_pyramid_mode": "mask3d_fpn"},
        "train": {"batch_scenes_per_step": 5, "gradient_accumulation_steps": 1, "drop_last": False,
                  "backbone_lr": 1e-4, "decoder_lr": 1e-4, "weight_decay": .01, "grad_clip_norm": None,
                  "backbone_freeze_epochs": 0, "deterministic_epoch_seed": True,
                  "persistent_workers": False, "save_latest_each_epoch": True},
        "eval": {"report_train_metrics": False, "report_dbscan": False,
                 "backbone_batchnorm_policy": "train_no_update", "topk_per_image": 100},
        "loss": {"include_auxiliary": True},
    }
    for section, settings in checks.items():
        for key, expected in settings.items():
            if config[section].get(key) != expected:
                raise ValueError(f"LA protocol requires {section}.{key}={expected!r}")
    if config["initialization"].get("freeze_backbone") or config["eval"].get("train_eval_split"):
        raise ValueError("LA requires full finetuning and forbids dense training evaluation")
    if config["data"].get("train_eval_split") or train.get("normalization_target_instances") is not None:
        raise ValueError("LA must not derive target counts or losses from dense training labels")
    scheduler = config.get("scheduler", {})
    defaults = dict(name="onecyclelr", interval="optimizer_step", pct_start=.3,
                    anneal_strategy="cos", cycle_momentum=True, base_momentum=.85,
                    max_momentum=.95, div_factor=25., final_div_factor=1e4, three_phase=False)
    if any(scheduler.get(key, value) != value for key, value in defaults.items()):
        raise ValueError("LA requires the declared OneCycleLR shape")
    if scheduler.get("max_lr") is not None and scheduler["max_lr"] not in (1e-4, [1e-4, 1e-4]):
        raise ValueError("LA encoder and decoder OneCycleLR peaks must both be 1e-4")


def suppress_empty_adamw_update(optimizer, criterion):
    """Clear gradients before step: AdamW must not decay a zero-observation batch."""
    observed = any(len(rows) for rows, _ in criterion.all_matches[0])
    if not observed:
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                parameter.grad = None
    return observed


def validate_voxel_mapping(predictions):
    import torch
    for prediction in predictions:
        if not torch.equal(prediction["point_to_segment"], prediction["point_to_voxel"].long()):
            raise ValueError("LA mask domain differs from the actual backbone voxel mapping")


class SupervisionAudit:
    def __init__(self):
        self.digest = hashlib.sha256()
        self.scenes = []
        self.empty_batches = 0
        self.effective_updates = 0

    def observe(self, batch, targets):
        self.scenes.extend(batch.scene_ids)
        for value in (batch.points, batch.features,
                      *(v for t in targets for v in (t.vertex_indices, t.instance_ids, t.semantic_ids))):
            self.digest.update(value.detach().cpu().contiguous().numpy().tobytes())

    def finish(self, expected_scenes, criterion, epoch):
        if sorted(self.scenes) != sorted(expected_scenes):
            raise ValueError("LA epoch did not visit each selected training scene exactly once")
        rows = criterion.statistics
        totals = {key: sum(row[key] for row in rows)
                  for key, value in rows[0].items() if isinstance(value, int)} if rows else {}
        criterion.statistics.clear()
        return dict(epoch=epoch, scenes=len(self.scenes), supervision_sha256=self.digest.hexdigest(),
                    retention=totals, unknown_instances=None,
                    empty_supervision_batches=self.empty_batches,
                    effective_optimizer_updates=self.effective_updates,
                    scheduled_optimizer_slots=self.empty_batches + self.effective_updates)
