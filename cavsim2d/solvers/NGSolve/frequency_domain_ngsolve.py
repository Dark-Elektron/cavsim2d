"""Driven frequency-domain solve with waveguide-port boundaries (m = 0).

**This is a driven sweep, not an eigensolver.** At each frequency it assembles one
linear system, applies the port transparent boundary condition with ``w`` frozen, and
returns the longitudinal beam-coupling impedance

    Z(w) = - integral E_z(z, r=0) exp(+i w z / c0) dz            [Ohm]

For ``Q_ext`` per mode, use :mod:`eigen_ports` instead: there the port operator goes
INTO the eigenproblem, which makes it nonlinear in ``w`` and yields a complex
eigenfrequency whose imaginary part is the radiation loss. This module cannot give
``Q_ext`` except by fitting resonance widths, which is unreliable where poles overlap.

Intended growth: excite the ports themselves rather than a source current, and report
S- and Z-parameters — the role CST's frequency-domain solver plays. The port operators
it needs already live in :mod:`ports`.

The formulation
---------------
The eigen weak form (:meth:`NGSolveMEVP._build_system`) assembles

    stiff . E = k0^2 mass . E,      k0 = w / c0

with the ``2 pi`` and ``mu0`` scaling absorbed. The driven system adds the port term::

    [stiff - k0^2 mass + sum_n c_n(w) P_n P_n^T] E = 0,     c_n = i k0^2 / beta_n

``stiff`` and ``mass`` carry no frequency, so they are assembled ONCE and the sweep
only rebuilds the rank-1 port block. That is why the port is a separate assembled
block and not an extra term in the eigen weak form, which would force a full
re-assembly per frequency.

The excitation, and why it is not a source term
----------------------------------------------
There is no beam here, only a prescribed source current at exactly ``v = c``. Together
with the pipe around it that current forms a **coaxial line**, so its own field is the
TEM mode

    E_r = Z0 I / (2 pi r),      E_z = 0,      travelling as exp(-i k0 z)

an *exact* solution: it satisfies PEC on a uniform pipe wall, radiates nothing, and
having ``E_z = 0`` contributes nothing to ``Z``. That is the same statement as "a
smooth pipe has no impedance".

So the unknown is the **scattered** field. Write ``E = E_inc + E_sc``; ``E_sc`` carries
no source at all and is driven only by ``E_inc`` failing the PEC condition where the
wall departs from a uniform pipe. ``E_sc`` is regular everywhere -- including on the
axis, where ``E_inc`` is not -- and ``Z = - integral E_sc,z exp(+i k0 z) dz``.

Putting the current in as a source term instead does not work: ``Z`` then reads ``E_z``
back exactly where the source sits, which for a line current is logarithmically
singular. ``Im Z`` diverged with mesh refinement (-5.79 / -8.06 / -11.67 kOhm at
h = 8/6/4 mm). Spreading the current over a radius ``a`` removed the divergence but
left ``Im Z`` depending on ``a`` (-2.52 / -1.44 / -0.69 for a = 1.5/2/3 mm), which at
``v = c`` it must not. Taking the singular part out analytically leaves nothing to
regularise, and both parts then converge to four figures.

Scope
-----
PEC walls, so the only loss is radiation out of the ports. Below the pipe cutoff a
trapped mode has infinite Q and ``Z`` has a genuine pole on the real axis: the sweep
blows up there, correctly.

KNOWN ISSUE: on a TESLA 3-cell this currently returns ``Re Z < 0`` at the ~7% level
(-0.0164 kOhm at 4000 MHz against |Z| = 0.22), which violates passivity. It is neither
a mesh effect (identical at h = 12/8/6/4) nor port-mode truncation (identical for
n = 1/3/6), and is unexplained. Do not trust this module until that is resolved.
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from ngsolve import (BND, CoefficientFunction, GridFunction,  # type: ignore
                     LinearForm, TaskManager, ds, exp, x, y)

import pandas as pd

from cavsim2d.solvers.NGSolve.eigen_ngsolve import NGSolveMEVP, mesh_h_metres
from cavsim2d.solvers.NGSolve.ports import (C0, EPS0, MU0, Z0, cutoffs,
                                            port_planes, projection_vectors,
                                            propagation_constant)
from cavsim2d.utils.printing import info, warning

def _to_scipy(mat):
    """NGSolve sparse matrix -> scipy CSR (complex)."""
    rows, cols, vals = mat.COO()
    n = mat.height
    return sp.csr_matrix((np.array(vals, dtype=complex),
                          (np.array(rows), np.array(cols))), shape=(n, n))


class DrivenPortSolver:
    """Beam-driven frequency sweep of an axisymmetric cavity with waveguide ports.

    The ports replace the closed (PMC) ends of the beam pipes. PMC is the *natural*
    condition of this weak form, so the port enters purely as a boundary term: no
    constraint is removed and no interior equation changes.
    """

    #: Boundary name carrying the pipe ends when the cavity is meshed closed.
    PORT_BND = 'PMC'

    #: Radius over which the source current is spread [m]. A regularisation, not
    #: a physical size: at v = c the space-charge term vanishes, so Z must not
    #: depend on it. Keep it comfortably larger than the local mesh size and much
    #: smaller than the pipe.
    def __init__(self, n_port_modes=3):
        self.n_port_modes = int(n_port_modes)

    # -- geometry / port operators: see ports.py -------------------------
    def port_planes(self, mesh):
        return port_planes(mesh, self.PORT_BND)

    def port_vectors(self, mesh, fes):
        return projection_vectors(mesh, fes, self.n_port_modes, bnd=self.PORT_BND)

    @staticmethod
    def incident_field(k0):
        """The source's own field: the coax TEM mode, ``E_r = Z0/(2 pi r)``, ``E_z = 0``.

        Exact for a unit current at ``v = c`` on the axis of a uniform pipe. Written
        as an in-plane ``(E_z, E_r)`` CoefficientFunction to match the product space.
        """
        Z0 = MU0 * C0
        return CoefficientFunction((0.0, Z0 / (2 * np.pi * y))) * exp(-1j * k0 * x)

    def voltage_functional(self, mesh, fes, k0):
        """``integral v_z exp(+i k0 z) dz`` on the axis, as a vector.

        Applied to the SCATTERED field, which is regular there -- so unlike the
        source-term formulation this evaluates a smooth field, not a singular one.
        """
        v, _v_phi = fes.TestFunction()
        lf = LinearForm(fes)
        lf += (v.Trace()[0] * exp(1j * k0 * x)) * ds(definedon=mesh.Boundaries('AXI'))
        lf.Assemble()
        return np.array(lf.vec.FV().NumPy(), dtype=complex).copy()

    # -- the sweep --------------------------------------------------------
    def impedance(self, cav, freqs_mhz, mesh_h=12.0, mesh_p=3, beampipe_length=None):  # noqa: C901
        """Longitudinal beam impedance ``Z(f)`` [Ohm] over *freqs_mhz*.

        Returns a DataFrame with the same columns the rest of cavsim2d uses, so it
        overlays an eigenmode reconstruction or a wakefield run directly.
        """
        solver = NGSolveMEVP()
        h_m = mesh_h_metres({'h': mesh_h})
        cfg = {'mesh_config': {'h': mesh_h, 'p': mesh_p}}
        if beampipe_length is not None:
            cfg['beampipe_length'] = beampipe_length
        mesh = solver._build_mesh(cav, h_m, int(mesh_p),
                                  boundary_conditions=33, eigenmode_config=cfg)

        # The eigen weak form, unchanged, in complex arithmetic. Only 'stiff' and
        # 'mass' are taken -- they carry no frequency, so this happens once.
        system = solver._build_system(mesh, int(mesh_p), 0, complex_fes=True)
        fes, a, b = system['fes'], system['a'], system['b']
        with TaskManager():
            fes.Update()
            a.Assemble()
            b.Assemble()

        K, M = _to_scipy(a.mat), _to_scipy(b.mat)
        free = np.array([bool(d) for d in fes.FreeDofs()])
        idx = np.where(free)[0]
        dir_idx = np.where(~free)[0]              # PEC dofs, where the lift lives

        p_vecs, cutoffs = self.port_vectors(mesh, fes)
        info(f'driven port solve: {len(idx)} free dofs, {len(p_vecs)} port modes '
             f'({self.n_port_modes} per end), cutoffs '
             f'{np.unique(np.round(cutoffs * C0 / (2 * np.pi) * 1e-6, 1))} MHz')

        rows = []
        for f_mhz in np.atleast_1d(freqs_mhz).astype(float):
            k0 = 2 * np.pi * f_mhz * 1e6 / C0
            w = k0 * C0

            A_full = (K - (k0 ** 2) * M).astype(complex).tolil()
            # PORT: rank-1 per mode, over that port's dofs only
            for j, k_c in enumerate(cutoffs):
                beta = propagation_constant(k0, k_c)
                c_n = 1j * (k0 ** 2) / beta
                pj = p_vecs[j]
                nz = np.where(np.abs(pj) > 0)[0]
                if nz.size == 0:
                    continue
                A_full[np.ix_(nz, nz)] += c_n * np.outer(pj[nz], pj[nz])
            A_full = A_full.tocsc()
            A = A_full[idx][:, idx]

            # E_sc = -E_inc tangentially on PEC, and no source anywhere. The
            # boundary data is imposed as a lift: set the Dirichlet dofs, move the
            # resulting column action to the right-hand side, solve for the rest.
            gfu = GridFunction(fes)
            gfu.components[0].Set(-self.incident_field(k0), BND,
                                  definedon=mesh.Boundaries('PEC'))
            lift = np.array(gfu.vec.FV().NumPy(), dtype=complex).copy()
            rhs = -(A_full[idx][:, dir_idx] @ lift[dir_idx])
            try:
                sol = spla.spsolve(A, rhs)
            except Exception as exc:                       # near a trapped-mode pole
                warning(f'{f_mhz:.1f} MHz: solve failed ({exc}); Z set to nan')
                rows.append({'f [MHz]': f_mhz, 'Z': np.nan + 0j})
                continue
            e_sc = lift.copy()
            e_sc[idx] = sol

            volt = self.voltage_functional(mesh, fes, k0)
            rows.append({'f [MHz]': f_mhz, 'Z': -complex(volt @ e_sc)})

        df = pd.DataFrame(rows)
        z = df['Z'].to_numpy()
        return pd.DataFrame({'f [MHz]': df['f [MHz]'],
                             '|Z| [kOhm]': np.abs(z) * 1e-3,
                             'Re(Z) [kOhm]': z.real * 1e-3,
                             'Im(Z) [kOhm]': z.imag * 1e-3})
