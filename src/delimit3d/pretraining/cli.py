"""Public pretraining command, with the original strict trainer underneath."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import sys
import time


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    preview = argparse.ArgumentParser(
        description="Contrastive 3D pretraining from SAM2 or UnSAMv2 2D masks.",
        epilog="Additional trainer flags override recipe defaults. See docs/pretraining.md.",
    )
    preview.add_argument("--config", type=Path, required=True)
    preview.add_argument("--show-config", action="store_true", help="Print the merged recipe without loading a model.")
    preview.add_argument("--allow-shared-proposal-negatives", action=argparse.BooleanOptionalAction, default=None)
    preview.add_argument("--sam2-maximum-raw-points", type=int, choices=(100_000, 500_000))
    from delimit3d.training.sampling_ablation import VARIANTS
    preview.add_argument("--ablation-variant", choices=VARIANTS)
    preview.add_argument("--ablation-activation-update", type=int)
    preview.add_argument("--proposal-area-cutoff", type=int)
    preview.add_argument("--proposal-area-audit", type=Path)
    preview.add_argument("--source-root", type=Path)
    preview.add_argument("--train-split", type=Path)
    preview.add_argument("--holdout-split", type=Path)
    preview.add_argument("--litept-root", type=Path)
    preview.add_argument("--output-dir", type=Path)
    preview.add_argument("--initial-checkpoint", type=Path, help="Existing scratch initialization; otherwise create one beside the run.")
    preview.add_argument("--resume", type=Path)
    preview.add_argument("--resume-sha256")
    known, _ = preview.parse_known_args(argv)
    if known.show_config:
        from delimit3d.runtime.config import load_config
        config = load_config(known.config, expand_env=False)
        for key, value in vars(known).items():
            if key not in {"config", "show_config"} and value is not None:
                config[key] = str(value) if isinstance(value, Path) else value
        print(json.dumps(config, indent=2))
        return

    from .arguments import parse_args
    args = parse_args(argv)
    if args.initial_checkpoint is None:
        # The trainer creates output_dir exclusively. Initialization lives next
        # to it so that creating a fresh run cannot be mistaken for a resume.
        args.initial_checkpoint = args.output_dir.with_name(args.output_dir.name + "_initial.pt")
        if int(os.environ.get("RANK", "0")) == 0 and not args.initial_checkpoint.exists():
            from delimit3d.cli.initialize_scratch import create_initialization
            create_initialization(argparse.Namespace(
                litept_root=args.litept_root, litept_variant=args.litept_variant,
                output=args.initial_checkpoint,
                seed=args.seed if args.initialization_seed is None else args.initialization_seed,
                grid_size=args.grid_size, input_features=args.input_features,
                multiscale_supervision=args.multiscale_supervision,
                multiscale_loss_weight=args.multiscale_loss_weight,
                multiscale_warmup_epochs=args.multiscale_warmup_epochs,
            ))
        deadline = time.monotonic() + 600
        while not args.initial_checkpoint.is_file():
            if time.monotonic() >= deadline:
                raise TimeoutError("Rank zero did not finish creating the scratch initialization")
            time.sleep(0.2)
    elif not args.initial_checkpoint.is_file():
        raise FileNotFoundError(args.initial_checkpoint)
    from delimit3d.runtime.config import expand_environment, load_config
    recipe = load_config(known.config, expand_env=False).get("recipe", {})
    if recipe.get("calibration_required"):
        from .calibration import validate_training_calibration
        report = args.calibration_report or Path(expand_environment(recipe["calibration_report"]))
        args.calibration_record = validate_training_calibration(report, args)
    from .experiment_record import prepare_execution_record
    prepare_execution_record(args)
    from .trainer import _run
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    _run(args)


if __name__ == "__main__":
    main()
