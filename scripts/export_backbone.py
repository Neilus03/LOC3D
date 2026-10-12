#!/usr/bin/env python3
"""Strict export of a pretrained native LitePT-S* backbone."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from delimit3d.pretraining.export import main
if __name__ == "__main__":
    main()
