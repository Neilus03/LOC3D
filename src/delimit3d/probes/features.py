"""Native feature extraction with an exact map back to original point rows."""
from __future__ import annotations
from contextlib import nullcontext
from pathlib import Path
import numpy as np
import torch

from delimit3d.models.backbones import backbone_name, build_ptv3_backbone
from delimit3d.models.litept_wrapper import LitePTBackbone, LitePTBackboneOutput
from delimit3d.tasks.agile3d.engine import load_frozen_backbone
from delimit3d.evaluation.backbone_batchnorm import current_batch_without_updates


def extract_native_features(config, points, colors, normals, *, device="cuda"):
    points, colors, normals = (np.asarray(value, dtype=np.float32) for value in (points, colors, normals))
    if points.ndim != 2 or points.shape[1] != 3 or colors.shape != points.shape or normals.shape != points.shape:
        raise ValueError("Expected ordered points, colors and normals, each [N,3]")
    if not len(points) or not all(np.isfinite(value).all() for value in (points, colors, normals)):
        raise ValueError("Scene arrays must be finite and nonempty")
    if colors.max() > 1:
        colors = colors / 255
    policy = config.get("probe", {}).get("coordinate_policy", "center_xy_min_z")
    shift = np.zeros(3, dtype=np.float32)
    if policy == "center_xy_min_z":
        shift[:2] = (points.min(0)[:2] + points.max(0)[:2]) / 2
        shift[2] = points.min(0)[2]
    elif policy == "per_axis_minimum":
        shift = points.min(0)
    elif policy != "native":
        raise ValueError("Unknown feature probe coordinate policy")
    name = backbone_name(config)
    grid = float(config["probe"]["grid_size"])
    if name == "litept_s_star":
        model = LitePTBackbone(litept_root=config["model"]["litept_root"], in_channels=6,
            grid_size=grid, litept_variant="litept_s_star", voxel_reduce="representative",
            representative_sampling="first", cache_training_voxelization=False)
        load_report = load_frozen_backbone(model, Path(config["probe"]["checkpoint"]))
    else:
        model = build_ptv3_backbone(config, frozen=True, grid_size=grid)
        load_report = model.load_report
    model.to(device).eval()
    coord = torch.as_tensor(points - shift, device=device)
    features = torch.as_tensor(np.concatenate((colors, normals), axis=1), device=device)
    bn_policy = config["probe"].get("backbone_batchnorm_policy", "eval")
    if bn_policy not in {"eval", "train_no_update"}:
        raise ValueError("Unknown feature probe BatchNorm policy")
    context = current_batch_without_updates(model) if bn_policy == "train_no_update" else nullcontext()
    with torch.no_grad(), context:
        output = model(coord, features)
    if isinstance(output, LitePTBackboneOutput):
        native, inverse = output.scene_tokens, output.inverse_map
        xyz = output.scene_xyz
    else:
        native, inverse, xyz = output.finest_features, output.original_to_finest, output.finest_xyz
    if inverse.shape != (len(points),) or inverse.min() < 0 or inverse.max() >= len(native):
        raise ValueError("Native features have an invalid original-point map")
    return native, inverse, {
        "backbone": name, "native_feature_dim": native.shape[1], "native_tokens": len(native),
        "original_points": len(points), "grid_size": grid, "coordinate_policy": policy,
        "shift": shift.tolist(), "backbone_batchnorm_policy": bn_policy, "checkpoint": load_report,
        "input_pipeline": "ordered_RGBN6_representative_downstream_bridge",
        "task_adapters_used": False,
    }
