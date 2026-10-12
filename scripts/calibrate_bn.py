#!/usr/bin/env python3
"""Create a separate checkpoint with recalibrated backbone BatchNorm buffers."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from delimit3d.cli.recalibrate_bn import main
if __name__ == "__main__":
    main()
