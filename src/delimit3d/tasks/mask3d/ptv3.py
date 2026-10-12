"""PTv3 Mask3D input bridge; the shared decoder is loaded before projections."""
from __future__ import annotations
import torch
from torch import nn
from delimit3d.models.concerto_wrapper import ConcertoBackbone, ConcertoMaskAdapters
from delimit3d.training.concerto_contract import (concerto_feature_recipe, concerto_input_projection, DIRECT_INPUT_PROJECTION)
from delimit3d.models.backbones import build_ptv3_backbone, ptv3_recipe, ptv3_direct_inputs

def ptv3_model_type(config, instances):
    from delimit3d.tasks.mask3d.model import (
        LitePTMask3DModel, LitePTMask3DDecoder, split_feature_pyramid,
    )
    class ConcertoMask3D(LitePTMask3DModel):
        def __init__(self, *, litept_root, backbone_frozen=False, backbone_kwargs=None, decoder_kwargs=None):
            nn.Module.__init__(self)
            if backbone_kwargs["in_channels"] != 6:
                raise ValueError("Changed downstream voxel/input contract")
            feature_recipe = ptv3_recipe(config)
            self.direct_inputs = ptv3_direct_inputs(config)
            self.backbone = build_ptv3_backbone(
                config, frozen=config["initialization"]["freeze_backbone"],
                grid_size=float(backbone_kwargs["grid_size"]))
            self.adapters = (nn.Identity() if self.direct_inputs else
                             ConcertoMaskAdapters(seed=42, feature_recipe=feature_recipe))
            self.decoder = LitePTMask3DDecoder(memory_channels=(504,252,144,72), **decoder_kwargs)
            self._backbone_frozen = bool(config["initialization"]["freeze_backbone"])
            self.gradient_audit = {name: {} for name in ("backbone", "adapters", "decoder")}
            instances.append(self)

        def install_gradient_audit(self):
            # Install after input projections are replaced, so their actual
            # tensors (rather than the discarded template inputs) are audited.
            for component in self.gradient_audit:
                for name, parameter in getattr(self, component).named_parameters():
                    if parameter.requires_grad and config.get("_audit_gradients", False):
                        def capture(grad, component=component, name=name):
                            if not torch.isfinite(grad).all():
                                raise RuntimeError(f"Nonfinite gradient: {component}.{name}")
                            self.gradient_audit[component][name] = float(grad.detach().float().norm())
                            return grad
                        parameter.register_hook(capture)

        def set_backbone_frozen(self, frozen):
            self._backbone_frozen = bool(frozen)
            self.backbone.set_frozen(frozen)

        def parameter_groups(self, *, backbone_lr, decoder_lr, include_frozen_backbone=False):
            if include_frozen_backbone:
                raise ValueError("These comparisons do not use a staged unfreeze")
            groups = []
            if not self.backbone_frozen:
                groups.append({"name": "backbone", "lr": backbone_lr, "params": list(self.backbone.parameters())})
            groups.append({"name": "decoder", "lr": decoder_lr,
                           "params": [*self.adapters.parameters(), *self.decoder.parameters()]})
            return groups

        def forward(self, points, features, scan_segments, *, scene_offsets=None, return_debug=False):
            if points.ndim != 2 or points.shape[1] != 3 or features.shape != (len(points), 6):
                raise ValueError("Expected points [N,3] and RGBN6 features [N,6]")
            if scan_segments.shape != (len(points),):
                raise ValueError("One segment id is required per input point")
            offsets = self._validate_offsets(scene_offsets, len(points), points.device)
            # The pure backbone enforces no_grad itself. Adapters remain outside
            # that context, including for the frozen comparison.
            pyramid = self.adapters(self.backbone(points, features, point_offsets=offsets))
            scene_pyramids = split_feature_pyramid(pyramid)
            starts, ends = [0] + offsets.tolist()[:-1], offsets.tolist()
            outputs = []
            for index, (scene, start, end) in enumerate(zip(scene_pyramids, starts, ends, strict=True)):
                segment_ids, point_to_segment = torch.unique(scan_segments[start:end].long(), sorted=True, return_inverse=True)
                voxel_to_segment = point_to_segment[scene.representative_indices.long()]
                prediction = self.decoder(scene, point_to_segment=point_to_segment,
                    voxel_to_segment=voxel_to_segment, num_segments=int(segment_ids.numel()), return_debug=return_debug)
                prediction.update(scene_index=index, segment_ids=segment_ids,
                    point_to_segment=point_to_segment, voxel_to_segment=voxel_to_segment,
                    point_to_voxel=scene.original_to_finest, representative_indices=scene.representative_indices)
                for auxiliary in prediction["aux_outputs"]:
                    auxiliary["mask_logits"] = auxiliary["segment_mask_logits"][:, point_to_segment.long()]
                    auxiliary["voxel_mask_logits"] = auxiliary["segment_mask_logits"][:, voxel_to_segment.long()]
                outputs.append(prediction)
            return outputs
    return ConcertoMask3D
