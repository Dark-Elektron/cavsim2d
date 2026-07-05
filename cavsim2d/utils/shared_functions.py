"""Backward-compatible re-exports from split utility modules.

This module was split into:
- cavsim2d.utils.geometry     (geometry creation and manipulation)
- cavsim2d.utils.quadrature   (quadrature and sampling utilities)
- cavsim2d.utils.data_utils   (data processing and file I/O)

All symbols are re-exported here for backward compatibility.
"""
from cavsim2d.utils.geometry import *
from cavsim2d.utils.quadrature import *
from cavsim2d.utils.data_utils import *
from cavsim2d.utils.printing import *
