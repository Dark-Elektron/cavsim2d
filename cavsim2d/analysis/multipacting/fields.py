"""Field adapter: cavsim2d eigenmode result -> multipacting EM field.

The particle integrator evaluates ``em.e(mesh_points)`` (in-plane E, a 2-vector
(E_z, E_r)) and ``em.h(mesh_points)`` (azimuthal H, scalar) directly on the
GridFunctions. cavsim2d's eigenmode field is a **product space** ``(u, u_phi)``,
so this adapter:

- takes E from the in-plane HCurl block, ``gfu_E[mode].components[0]``;
- **reconstructs** H as ``1j/(mu0*w) * curl(E_inplane)``.

The reconstruction is deliberate. cavsim2d stores a magnitude-only ``H_phi``
(the ``1j`` dropped, since only |H| enters the QOIs), but multipacting integrates
E and B together over the RF cycle and needs the physical 90-degree E-H phase
that the ``1j`` carries. Rebuilding H from ``curl(E)`` reproduces the field
PyMultipact used exactly.
"""
import json
import os
import pickle

import numpy as np
from ngsolve import curl

from cavsim2d.solvers.NGSolve.eigen_ngsolve import NGSolveMEVP, mesh_h_metres
from cavsim2d.solvers.eigenmode_result import pol_number

mu0 = 4 * np.pi * 1e-7


class EMField:
    """Holder whose ``.e`` / ``.h`` are coefficient functions the integrator
    evaluates directly at mesh points (``em.e(mesh(z, r))``)."""

    def __init__(self, e, h):
        self.e = e
        self.h = h


def build_emfield(gfu_E, mode, freq_mhz):
    """In-plane E and reconstructed azimuthal H for a monopole product-space mode.

    Parameters
    ----------
    gfu_E : list
        The eigenmode E fields (product-space GridFunctions) from ``gfu_EH.pkl``.
    mode : int
        Mode index.
    freq_mhz : float
        Mode frequency [MHz], for the reconstruction weight ``w = 2*pi*f``.
    """
    w = 2 * np.pi * freq_mhz * 1e6
    e_inplane = gfu_E[mode].components[0]         # HCurl block: (E_z, E_r)
    h_phi = 1j / (mu0 * w) * curl(e_inplane)      # physical azimuthal H
    return EMField(e_inplane, h_phi)


def load_eigenmode_fields(fields_dir):
    """Load ``(mesh, gfu_E, gfu_H)`` from an eigenmode polarisation folder.

    Reads the ``mesh.pkl`` and ``gfu_EH.pkl`` the eigenmode solver writes into
    ``eigenmode/<pol>/``. Kept separate from :func:`build_emfield` so a worker
    process can load once and build the field per swept mode.
    """
    with open(os.path.join(fields_dir, 'mesh.pkl'), 'rb') as f:
        mesh = pickle.load(f)
    with open(os.path.join(fields_dir, 'gfu_EH.pkl'), 'rb') as f:
        gfu_E, gfu_H = pickle.load(f)
    return mesh, gfu_E, gfu_H


def solve_multipacting_field(cav, save_dir, mesh_config=None, n_modes=None,
                             polarisation='monopole'):
    """Solve the eigenmode on a multipacting-specific mesh, into the
    multipacting analysis' OWN field folder.

    Used when a custom surface refinement (``pec_maxh``) is requested: the
    surviving electrons live in a sub-millimetre layer at the wall, so
    multipacting needs a finer PEC surface than the eigenmode QOIs do. Solving
    on a separate mesh here leaves ``cav.eigenmode``'s results (and everything
    downstream of them) completely untouched — each analysis stays internally
    consistent with its own mesh.

    *mesh_config* takes ``h`` / ``p`` (global size [mm] / order, as in the
    eigenmode config) plus ``pec_maxh`` (local PEC surface size [mm]).
    *polarisation* selects the azimuthal order to solve ('monopole', 'dipole', ...)
    and *n_modes* how many modes of it — so multipacting can be driven by, say, the
    second monopole mode or a dipole mode. Writes
    ``mesh.pkl`` + ``gfu_EH.pkl`` + ``freqs.json`` into *save_dir* (the same
    layout as an eigenmode polarisation folder, so the sweep driver reads it
    unchanged) and returns the mode-frequency list [MHz].
    """
    cfg = dict(mesh_config or {})
    maxh = mesh_h_metres(cfg, default=6)
    order = cfg.get('p', 3)
    pec_maxh = cfg.get('pec_maxh')
    edge_maxh = {'PEC': float(pec_maxh) * 1e-3} if pec_maxh else None

    maker = getattr(cav, 'profile', None)
    profile = maker() if callable(maker) else None
    if profile is None:
        raise RuntimeError(
            f"{type(cav).__name__} {cav.name!r} has no profile(), so a "
            f"multipacting-specific (PEC-refined) mesh cannot be built. Run "
            f"without pec_maxh to reuse the eigenmode field instead.")

    solver = NGSolveMEVP()
    n = solver.requested_n_modes(cav, None, n_modes)
    # The mesh is deliberately STRAIGHT (order=1 geometry, no Curve): the
    # tracker's collision surface is the polyline through the wall vertices,
    # and on a curved mesh that polyline disagrees with the element boundary by
    # the chord sagitta — impact points computed on the polyline then fall
    # OUTSIDE the curved mesh and the field evaluation throws ("Meshpoint not
    # in mesh"), first at concave wall sections. A straight mesh makes the
    # collision polyline coincide with the element edges exactly — the same
    # construction PyMultipact validated (polygon boundary + order-3 fields).
    # With the fine pec_maxh wall the geometric error is the chord sagitta
    # (~h^2/8R, micrometres); the field itself keeps the full order *p*.
    mesh = profile.mesh(maxh=maxh, order=1, edge_maxh=edge_maxh)
    system = solver._build_system(mesh, order, pol_number(polarisation))
    freqs, gfu_E, gfu_H = solver._solve_system(system, n)

    os.makedirs(save_dir, exist_ok=True)
    solver.save_mesh(save_dir, mesh)
    solver.save_fields(save_dir, gfu_E, gfu_H)
    freqs = [float(f) for f in freqs]
    with open(os.path.join(save_dir, 'freqs.json'), 'w') as f:
        json.dump(freqs, f, indent=4)
    return freqs
