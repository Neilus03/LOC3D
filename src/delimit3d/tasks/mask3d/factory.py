"""Task input adapters and initialization chosen without replacing engine globals."""
from __future__ import annotations

from delimit3d.models.backbones import backbone_name, ptv3_recipe
from delimit3d.models.concerto_wrapper import configure_direct_decoder_inputs, tensor_hash, component_counts
from .initialization import initialize_litept_mask3d_arm, load_decoder_template, load_litept_backbone_checkpoint
from delimit3d.runtime.records import file_sha256
from .model import LitePTMask3DModel
from .ptv3 import ptv3_model_type


def build_model(config, device):
    settings, initialization = config["model"], config["initialization"]
    name = backbone_name(config)
    constructor = LitePTMask3DModel if name == "litept_s_star" else ptv3_model_type(config, [])
    model = constructor(
        litept_root=settings.get("litept_root"),
        backbone_frozen=False,
        backbone_kwargs={
            "in_channels": int(settings.get("input_channels", 6)),
            "grid_size": float(settings.get("voxel_size", 0.05)),
            "litept_variant": "litept_s_star",
            "feature_pyramid_mode": settings.get("feature_pyramid_mode", "mask3d_fpn"),
            "voxel_reduce": settings.get("voxel_reduce", "representative"),
            "multi_scale": False,
            "representative_sampling": settings.get("representative_sampling", "first"),
        },
        decoder_kwargs={
            "train_queries": int(settings.get("train_queries", 100)),
            "eval_queries": int(settings.get("eval_queries", 100)),
            "num_classes": int(settings.get("num_classes", 2)),
            "mask_only": bool(settings.get("mask_only", False)),
        },
    ).to(device)
    if name == "litept_s_star":
        report = initialize_litept_mask3d_arm(
            model, decoder_template=initialization["decoder_template"],
            backbone_source=initialization["backbone_source"],
            backbone_checkpoint=initialization.get("backbone_checkpoint"),
            freeze_backbone=initialization.get("freeze_backbone", False),
        )
        if initialization.get("scratch_checkpoint"):
            if initialization["backbone_source"] != "scratch":
                raise ValueError("scratch_checkpoint is only valid for the scratch arm")
            report["backbone_checkpoint"] = load_litept_backbone_checkpoint(model, initialization["scratch_checkpoint"])
            report["backbone_state_sha256"] = model.backbone_state_hash()
    else:
        # Load the original 72D decoder strictly before changing input widths.
        decoder = load_decoder_template(model, initialization["decoder_template"])
        projection = None
        if model.direct_inputs:
            projection = configure_direct_decoder_inputs(
                model.decoder, task="mask3d", feature_recipe=ptv3_recipe(config),
                seed=config[name].get("adapter_seed", config["experiment"].get("seed", 42)),
            )
        model.install_gradient_audit()
        report = {
            "backbone_source": name, "backbone_frozen": model.backbone_frozen,
            "backbone_state_sha256": model.backbone_state_hash(),
            "decoder_state_sha256": model.decoder_state_hash(),
            "adapter_state_sha256": tensor_hash(model.adapters),
            "decoder_template": decoder, "backbone_checkpoint": model.backbone.load_report,
            "projection_initialization": projection, "parameters": component_counts(model),
            "official_mask3d_weights_loaded": False,
        }
    report["source_file_sha256"] = {
        key: file_sha256(initialization[key])
        for key in ("decoder_template", "backbone_checkpoint", "scratch_checkpoint")
        if initialization.get(key)
    }
    return model, report
