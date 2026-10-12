"""Single task model with explicit LitePT and native PTv3 feature interfaces."""
from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
import torch
from torch import nn

from delimit3d.evaluation.agile3d_decoder import load_initialization, state_sha256
from delimit3d.evaluation.backbone_batchnorm import normalized_encoder_forward
from delimit3d.models.backbones import backbone_name, build_ptv3_backbone, ptv3_recipe, ptv3_direct_inputs
from delimit3d.models.concerto_wrapper import agile_adapter, configure_direct_decoder_inputs, tensor_hash
from delimit3d.models.litept_wrapper import LitePTBackbone, LitePTBackboneOutput
from .engine import load_frozen_backbone


class Agile3DModel(nn.Module):
    def __init__(self, config, paths):
        super().__init__()
        settings = config["model"]
        name = backbone_name(config)
        self.frozen_encoder = bool(settings.get("freeze_encoder", False))
        self.eval_backbone_batchnorm_policy = config["evaluation"]["backbone_batchnorm_policy"]
        grid = float(config["protocol"]["voxel_size_m"])
        projection_report = None
        if name == "litept_s_star":
            initialization = config["encoder_initialization"]
            if initialization.get("kind") not in {"scratch", "pretrained_checkpoint"}:
                raise ValueError("encoder_initialization.kind must be scratch or pretrained_checkpoint")
            checkpoint = initialization.get("checkpoint")
            if bool(checkpoint) != (initialization["kind"] == "pretrained_checkpoint"):
                raise ValueError("Scratch initialization requires checkpoint=null; pretrained initialization requires a checkpoint")
            self.backbone = LitePTBackbone(
                litept_root=str(paths["litept_root"]), in_channels=6, grid_size=grid,
                litept_variant="litept_s_star", multi_scale=False,
                cache_training_voxelization=False, feature_pyramid_mode=None,
                voxel_reduce="representative", representative_sampling="first",
            )
            self.encoder_checkpoint_report = (load_frozen_backbone(self.backbone, Path(checkpoint))
                if checkpoint else {"kind": "scratch", "seed": config["seed"]})
            self.adapters = nn.Identity()
            native_width = 72
        else:
            self.backbone = build_ptv3_backbone(config, frozen=self.frozen_encoder, grid_size=grid)
            self.encoder_checkpoint_report = self.backbone.load_report
            self.adapters = (nn.Identity() if ptv3_direct_inputs(config) else
                             agile_adapter(config["seed"], feature_recipe=ptv3_recipe(config)))
            native_width = self.backbone.feature_channels[0] if ptv3_direct_inputs(config) else 72
        kwargs = {key: settings[key] for key in (
            "hidden_dim", "num_heads", "dim_feedforward", "num_decoders", "num_bg_queries",
            "dropout", "pre_norm", "max_click_events", "normalize_pos_enc", "gauss_scale", "aux")}
        # All models share the exact 72D fresh decoder template. Only input
        # projections change for the direct 992D/896D PTv3 interfaces.
        kwargs["feature_dim"] = 72
        self.decoder, self.decoder_initialization_report = load_initialization(
            paths["decoder_initialization"], expected_kwargs=kwargs)
        if name != "litept_s_star" and ptv3_direct_inputs(config):
            projection_report = configure_direct_decoder_inputs(
                self.decoder, task="agile3d", feature_recipe=ptv3_recipe(config),
                seed=config[name].get("adapter_seed", config["seed"]))
        if int(settings.get("feature_dim", native_width)) != native_width:
            raise ValueError("Configured decoder input width differs from the selected readout")
        if self.frozen_encoder:
            self.backbone.requires_grad_(False)
            self.backbone.eval()
        self.initial_encoder_state_sha256 = state_sha256(self.backbone.state_dict())
        self.initialization = {
            "backbone": name, "encoder": self.encoder_checkpoint_report,
            "decoder": self.decoder_initialization_report,
            "projection_initialization": projection_report,
            "adapter_state_sha256": tensor_hash(self.adapters), "native_feature_dim": native_width,
            "frozen_encoder": self.frozen_encoder,
        }

    def train(self, mode=True):
        super().train(mode)
        if self.frozen_encoder:
            self.backbone.eval()
        return self

    @normalized_encoder_forward
    def encode_batch(self, grid_coord, raw_coord, features, point_offsets):
        if self.frozen_encoder and any(module.training for module in self.backbone.modules()):
            raise RuntimeError("Frozen encoder normalization layer or stochastic layer entered training mode")
        with torch.no_grad() if self.frozen_encoder else nullcontext():
            output = self.backbone(raw_coord, features, point_offsets=point_offsets, grid_coord=grid_coord)
        if isinstance(output, LitePTBackboneOutput):
            tokens, xyz, offsets = output.scene_tokens, output.scene_xyz, output.scene_token_offsets
        else:
            tokens, xyz, offsets = output.finest_features, output.finest_xyz, output.finest_offsets
        tokens = self.adapters(tokens)
        ends = offsets.tolist()
        starts = [0] + ends[:-1]
        return ([tokens[a:b] for a, b in zip(starts, ends, strict=True)],
                [xyz[a:b] for a, b in zip(starts, ends, strict=True)])

    def decode(self, scene_tokens, scene_xyz, clicks, click_times):
        return self.decoder(scene_tokens, scene_xyz, clicks=clicks, click_times=click_times)
