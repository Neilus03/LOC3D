#!/usr/bin/env python3
"""Cosine or spherical k-means segmentation of native features."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from delimit3d.probes.cli import main
if __name__ == "__main__":
    main()
