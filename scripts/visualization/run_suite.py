#!/usr/bin/env python3
"""Run the Delimit3D visual atlas without requiring an editable install."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from delimit3d.visualization.suite import main

if __name__ == "__main__":
    raise SystemExit(main())
