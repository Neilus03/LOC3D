#!/usr/bin/env python3
"""Run trained-model inference; see --help."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from delimit3d.inference.cli import main

if __name__ == "__main__":
    main()
