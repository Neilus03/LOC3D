#!/usr/bin/env python3
"""Prepare geometry, 2D masks and immutable pretraining sources."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from delimit3d.data.cli import main
if __name__ == "__main__":
    main()
