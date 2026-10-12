"""Pinned NumPy quantization utilities used in the RGBN6 comparisons."""
from types import SimpleNamespace
from delimit3d.evaluation.agile3d_minkowski_compat import _sparse_quantize, _batched_coordinates
ME = SimpleNamespace(utils=SimpleNamespace(sparse_quantize=_sparse_quantize, batched_coordinates=_batched_coordinates))
