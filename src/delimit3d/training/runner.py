"""Compatibility entry point for the modular pretraining trainer."""

import sys
from delimit3d.pretraining import trainer as _trainer

if __name__ == "__main__":
    _trainer.main()
else:
    sys.modules[__name__] = _trainer
