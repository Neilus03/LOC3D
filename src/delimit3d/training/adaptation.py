"""Compatibility path for the modular pretraining API."""

import sys
from delimit3d.pretraining import api as _api

sys.modules[__name__] = _api
