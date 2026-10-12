"""Compatibility import for the single supported Mask3D implementation."""
import sys
from delimit3d.tasks.mask3d import model as _implementation
sys.modules[__name__] = _implementation
