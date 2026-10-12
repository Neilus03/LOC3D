"""Strict evaluation of earlier full models and frozen-decoder checkpoints."""
from pathlib import Path
from delimit3d.runtime.records import file_sha256
from .engine import cpu_state_dict, state_sha256


def evaluation_checkpoint(payload, model, path):
    schema = payload.get("schema", "")
    if not schema.startswith(("delimit3d_agile3d_", "delimit3d_concerto_agile3d/", "delimit3d_sonata_agile3d/")):
        raise ValueError("Unsupported historical AGILE3D checkpoint schema")
    if "epoch" not in payload:
        raise ValueError("Historical checkpoint has no epoch record")
    if "model" in payload:
        model.load_state_dict(payload["model"], strict=True)
    elif "decoder" in payload:
        if payload.get("encoder_state_sha256") != state_sha256(model.backbone.state_dict()):
            raise ValueError("Frozen decoder requires its exact historical encoder checkpoint")
        model.decoder.load_state_dict(payload["decoder"], strict=True)
    else:
        raise ValueError("Historical checkpoint has no model or decoder tensors")
    return {"model": cpu_state_dict(model), "epoch": payload["epoch"], "step": payload.get("step", 0),
            "historical_checkpoint": {"schema": schema, "sha256": file_sha256(Path(path)),
                                       "strict_tensor_load": True, "resume_state_converted": False}}
