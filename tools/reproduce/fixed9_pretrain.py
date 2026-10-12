#!/usr/bin/env python3
"""Historical fixed-nine pretraining entry point; requires its sealed artifacts."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from delimit3d.training.granularity_pf69_runner import main

if __name__ == "__main__":
    main()
