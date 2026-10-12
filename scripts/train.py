#!/usr/bin/env python3
"""Train Mask3D or AGILE3D from a task config."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from delimit3d.tasks.cli import train_main
if __name__ == "__main__":
    train_main()
