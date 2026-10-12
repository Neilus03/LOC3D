"""Explicit, stateless BatchNorm inference for the LitePT AGILE3D adapters."""
from contextlib import contextmanager
from functools import wraps
import hashlib
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


DEFAULT_POLICY = "train_no_update"
PROTOCOL = "agile3d_backbone_batchnorm/v1"


def configured_policy(config):
    policy = config.get("evaluation", {}).get("backbone_batchnorm_policy", DEFAULT_POLICY)
    if policy not in ("eval", "train_no_update"):
        raise ValueError(f"Unknown backbone BatchNorm evaluation policy: {policy!r}")
    return policy


def evaluation_metadata(model):
    policy = getattr(model, "eval_backbone_batchnorm_policy", "eval")
    return {
        "backbone_batchnorm_policy": policy,
        "normalization_protocol": PROTOCOL,
        "normalization_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "normalization_statistics": "current_scene" if policy == "train_no_update" else "saved_running",
        "normalization_buffers_updated": False,
        "stochastic_layers": "eval",
    }


@contextmanager
def current_batch_without_updates(backbone):
    """Use F.batch_norm without running buffers; preserve every module's mode.

    This is numerically the same normalization as BN training mode, but it
    cannot change running_mean, running_var or num_batches_tracked. Keeping the
    modules in eval also respects the frozen encoder's mode guards.
    """
    if any(module.training for module in backbone.modules()):
        raise RuntimeError("Inference normalization requires an entirely eval-mode backbone")
    saved = []
    try:
        for module in backbone.modules():
            if not isinstance(module, nn.modules.batchnorm._BatchNorm):
                continue
            previous = module.__dict__.get("forward")
            had_forward = "forward" in module.__dict__
            saved.append((module, had_forward, previous))

            def stateless_forward(input, bn=module):
                bn._check_input_dim(input)
                return F.batch_norm(input, None, None, bn.weight, bn.bias,
                                    training=True, momentum=0.0, eps=bn.eps)

            module.forward = stateless_forward
        yield
    finally:
        for module, had_forward, previous in reversed(saved):
            if had_forward:
                module.forward = previous
            else:
                del module.forward


def normalized_encoder_forward(function):
    """Leave optimization forwards intact; select BN policy only in root eval."""
    @wraps(function)
    def encode(model, *args, **kwargs):
        if model.training or model.eval_backbone_batchnorm_policy == "eval":
            return function(model, *args, **kwargs)
        with current_batch_without_updates(model.backbone):
            return function(model, *args, **kwargs)
    return encode
