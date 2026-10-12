#!/usr/bin/env python3
"""Evaluate a full Mask3D or AGILE3D checkpoint."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from delimit3d.tasks.cli import evaluate_main
if __name__ == "__main__":
    evaluate_main()
