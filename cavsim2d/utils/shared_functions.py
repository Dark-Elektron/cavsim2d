"""Backward-compatible re-exports from split utility modules.

Geometry now lives in its own package, :mod:`cavsim2d.geometry`:

- ``cavsim2d.geometry.tangency``          (alpha / tangent-point solvers)
- ``cavsim2d.geometry.primitives``        (line and arc point sampling)
- ``cavsim2d.geometry.plotting``          (matplotlib contour renderers)
- ``cavsim2d.geometry.writers.gmsh``      (``.geo`` writers)
- ``cavsim2d.geometry.writers.multipac``  (Multipac exporters)
- ``cavsim2d.geometry.writers.cst``       (CST parameter export)

The remaining utilities were split into:

- ``cavsim2d.utils.shapes``      (shape-dict continuity, perturbation, tabulation)
- ``cavsim2d.utils.quadrature``  (quadrature and sampling utilities)
- ``cavsim2d.utils.data_utils``  (data processing and file I/O)

All symbols are re-exported here for backward compatibility. New code should
import from the specific module instead.

``cavsim2d.geometry.writers.abci`` is deliberately absent: it imports
``abci_code``, which star-imports this module, so re-exporting it here would
create an import cycle.
"""
from cavsim2d.geometry.tangency import *
from cavsim2d.geometry.primitives import *
from cavsim2d.geometry.plotting import *
from cavsim2d.geometry.writers.gmsh import *
from cavsim2d.geometry.writers.multipac import *
from cavsim2d.geometry.writers.cst import *
from cavsim2d.utils.shapes import *
from cavsim2d.utils.quadrature import *
from cavsim2d.utils.data_utils import *
from cavsim2d.utils.printing import *
