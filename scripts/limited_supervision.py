#!/usr/bin/env python3
"""Prepare, verify, train and evaluate matched limited-supervision experiments."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from delimit3d.tasks.limited_supervision import main

if __name__ == "__main__":
    main()
