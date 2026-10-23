#!/usr/bin/env python3
"""Run LOC3D pretraining. See --help and configs/pretrain/."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from delimit3d.pretraining.cli import main

if __name__ == "__main__":
    main()
