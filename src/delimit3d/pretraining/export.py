"""Export the native LitePT-S* backbone, strictly dropping pretraining heads."""
from __future__ import annotations
import argparse
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--litept-root", type=Path, required=True)
    parser.add_argument("--grid-size", type=float, required=True, help="Checkpoint's input voxel size.")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)
    import torch
    from delimit3d.models.litept_wrapper import LitePTBackbone
    from delimit3d.tasks.agile3d.engine import load_frozen_backbone
    from delimit3d.runtime.records import file_sha256, save_torch, write_json
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = LitePTBackbone(litept_root=str(args.litept_root), in_channels=6,
        grid_size=args.grid_size, litept_variant="litept_s_star", voxel_reduce="representative",
        representative_sampling="first", cache_training_voxelization=False)
    report = load_frozen_backbone(model, args.checkpoint)
    state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    output = {
        "schema": "delimit3d_backbone/v1", "backbone_state_dict": state,
        "architecture": {"litept_variant": "litept_s_star", "input_features": "rgbn6",
                         "grid_size": args.grid_size, "native_dim": 72, "representative_sampling": "first"},
        "source_checkpoint_sha256": file_sha256(args.checkpoint),
        "epoch": payload.get("epoch"), "step": payload.get("step", payload.get("update")),
        "strict_load": report, "task_decoder_weights_included": False,
        "pretraining_heads_included": False,
    }
    save_torch(args.output, output)
    write_json(args.output.with_suffix(".json"), {key: value for key, value in output.items() if key != "backbone_state_dict"})
    print(f"Exported {len(state)} native backbone tensors to {args.output}")
