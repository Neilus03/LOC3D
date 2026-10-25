#!/usr/bin/env python3
"""Prepare or verify sealed official ScanNet low-annotation inputs."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from delimit3d.tasks.mask3d.prepare_low_annotation import main

if __name__ == "__main__":
    main()
