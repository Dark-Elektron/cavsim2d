import json
import functools
import gc
import os.path
import pickle
import sys
import time
import warnings

from matplotlib import tri
from cavsim2d.utils.shared_functions import *
from ngsolve import *
from ngsolve import (x, y, dx, pi, Mesh, exp, BND, # type: ignore
                     GridFunction, BilinearForm, InnerProduct, curl, grad, Conj, # type: ignore
                     Integrate, TaskManager, HCurl, H1, Preconditioner, solvers, Norm, # type: ignore
                     IdentityMatrix, ArnoldiSolver) # type: ignore
from ngsolve.la import Embedding # type: ignore
from ngsolve.webgui import Draw
from ngsolve.comp import VorB # type: ignore
from netgen.occ import *
from netgen.occ import OCCGeometry
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from cavsim2d.utils.printing import *
from cavsim2d.solvers.eigenmode_result import pol_name, pol_number
import gmsh
import platform

mu0 = 4 * pi * 1e-7
eps0 = 8.85418782e-12
c0 = 299792458
SIGMA_COPPER = 5.96e7  # electrical conductivity of copper [S/m]
DEFAULT_N_MODES = 10
# Loss tangent above which perturbation theory stops being trustworthy and the
# automatic loss model switches to the full complex eigenproblem: perturbation is
# O(tan_delta^2) in the eigenvalue and blind to field redistribution.
LOSSY_TAN_DELTA = 1e-2
# Krylov vectors per shift in the lossy (Arnoldi) solve. len(vecs) eigenpairs
# nearest the shift come back, from a Krylov space of 2*len(vecs)+1.
DEFAULT_ARNOLDI_VECTORS = 6
# Relative eigenpair residual below which an Arnoldi Ritz pair counts as a
# converged mode. Unconverged pairs come back looking like extra modes at
# plausible frequencies, so they are filtered on this rather than on a heuristic.
ARNOLDI_RESIDUAL_TOL = 1e-8
# After the last refinement pass the remaining pairs are judged on two looser
# bounds. Below WARN a mode is simply accurate enough to report without comment;
# between WARN and ACCEPT it is a real but loosely converged mode, worth saying so
# about; above ACCEPT it is a Krylov artefact and is dropped. Measured separation
# on this formulation: converged pairs sit at 1e-12 and below, real-but-loose ones
# at 1e-7..1e-4, artefacts at 1e-2 and above.
ARNOLDI_RESIDUAL_WARN = 1e-5
ARNOLDI_RESIDUAL_ACCEPT = 1e-3
# Shift-refinement passes. Pass 1 centres on the lossless eigenvalues; each further
# pass re-centres on what the previous one found, which is what a large tan_delta
# needs and a small one never triggers.
ARNOLDI_MAX_PASSES = 3
# Radius [m] below which a point counts as "on the axis". The 1/r weights in the
# azimuthal field are singular there but the fields are defined by their limit,
# so the weight is clamped rather than evaluated (the axis is measure-zero in
# every integral that uses it).
AXIS_EPS = 1e-9


def mesh_h_metres(mesh_config, default=20):
    """``mesh_config['h']`` (mm) -> metres, trapping the classic units slip.

    ``h`` is in **millimetres**. Passing metres — e.g. ``h=25e-3`` intending
    "25 mm" — asks for 25-*micron* elements: the mesher then grinds on millions
    of elements and the run looks hung. Warn loudly instead of silently melting
    the machine.
    """
    h = (mesh_config or {}).get('h', default)
    if h < 0.5:
        # warnings.warn (not the verbosity-gated warning()): a silent units
        # slip just looks like a hung run.
        warnings.warn(
            f"mesh_config['h'] is in MILLIMETRES: h={h:g} requests {h:g} mm "
            f"({h * 1e-3:.1e} m) elements — meshing will be extremely slow. "
            f"If you meant {h * 1e3:g} mm, pass h={h * 1e3:g}.",
            UserWarning, stacklevel=3)
    return h * 1e-3


def parse_materials(materials):
    """Validated ``{name: (eps_re, eps_im)}`` for a *materials* mapping.

    Each entry is a bare number or a dict::

        {'quartz': 3.8}                                  # lossless shorthand
        {'quartz': {'eps_r': 3.8}}
        {'quartz': {'eps_r': 3.8, 'tan_delta': 1e-4}}    # eps = 3.8*(1 - 1e-4j)
        {'quartz': 3.8 - 3.8e-4j}                        # complex shorthand

    The convention is ``exp(+j w t)``, so a lossy permittivity is
    ``eps_r = eps' - j eps''`` with ``tan_delta = eps''/eps' >= 0``. A *negative*
    loss tangent (gain) is rejected: it would come back as a negative Q and read
    as a solver bug rather than as the input error it is.

    ``mu_r`` is still refused rather than ignored — magnetic materials change the
    stiffness form, not just the mass form.
    """
    props = {}
    for name, entry in (materials or {}).items():
        if isinstance(entry, bool):
            raise ValueError(f"materials[{name!r}] must be a number or a dict, got {entry!r}.")
        if isinstance(entry, (int, float, complex)):
            entry = {'eps_r': entry}
        if not isinstance(entry, dict):
            raise ValueError(
                f"materials[{name!r}] must be a number or a dict like "
                f"{{'eps_r': 3.8, 'tan_delta': 1e-4}}, got {entry!r}.")
        unsupported = set(entry) - {'eps_r', 'tan_delta'}
        if unsupported:
            raise NotImplementedError(
                f"materials[{name!r}] sets {sorted(unsupported)}, which the eigenmode "
                "solver does not support yet — it models dielectrics (eps_r, tan_delta) "
                "only. Magnetic materials (mu_r) would change the weak form and the Q "
                "bookkeeping, so they are rejected rather than silently ignored.")
        eps = complex(entry.get('eps_r', 1.0))
        td = entry.get('tan_delta', None)
        if eps.imag and td is not None:
            raise ValueError(
                f"materials[{name!r}] gives BOTH a complex eps_r ({eps!r}) and a "
                f"tan_delta ({td!r}); they are two spellings of the same quantity. "
                "Pass one or the other.")
        if td is not None and float(td) < 0:
            raise ValueError(
                f"materials[{name!r}]['tan_delta'] must be >= 0 (a negative loss tangent "
                f"is gain, not loss), got {td!r}.")
        eps_re = eps.real
        if eps_re <= 0:
            raise ValueError(
                f"materials[{name!r}]: the real part of eps_r must be positive, "
                f"got {eps_re!r}.")
        # exp(+jwt) => eps = eps' - j eps''. A user writing 3.8+0.01j means the same
        # material as 3.8-0.01j, so the sign of the shorthand is not load-bearing.
        eps_im = abs(eps.imag) if eps.imag else eps_re * float(td or 0.0)
        props[name] = (eps_re, eps_im)
    return props


def _check_material_names(mesh, props):
    """Raise if any material name is not a region of *mesh*."""
    available = set(mesh.GetMaterials())
    for name in props:
        if name not in available:
            raise ValueError(
                f"materials names {name!r}, which is not a region of this mesh. "
                f"Available materials: {sorted(available)}. Add the region first "
                f"with cav.add_dielectric({name!r}, ...) — a name that does not match "
                "would otherwise solve the vacuum problem silently.")


def material_cfs(mesh, materials):
    """**Real** relative permittivity (eps') as a mesh-material-keyed CoefficientFunction.

    Returns ``None`` when there is nothing to weight — no materials given, or every
    one of them has eps' = 1. Callers **must** keep their plain vacuum expression in
    that case, so a single-domain cavity assembles exactly the forms it always did
    and cannot regress.

    This is the coefficient the *lossless* forms and the stored energy U use; the
    loss part is :func:`material_loss_cfs` and the two together are
    :func:`material_complex_cfs`. See :func:`parse_materials` for the accepted
    spellings.
    """
    props = parse_materials(materials)
    if not props:
        return None
    _check_material_names(mesh, props)
    eps = {name: re_ for name, (re_, _) in props.items()}
    if all(v == 1.0 for v in eps.values()):
        return None
    return mesh.MaterialCF(eps, default=1.0)


def material_loss_cfs(mesh, materials):
    """Imaginary relative permittivity (eps'' = eps' * tan_delta) as a
    CoefficientFunction, or ``None`` if every region is lossless.

    ``None`` is the signal that no dielectric-loss bookkeeping is needed at all, so
    a lossless run reports exactly the QOIs it always did.
    """
    props = parse_materials(materials)
    if not props:
        return None
    _check_material_names(mesh, props)
    loss = {name: im for name, (_, im) in props.items()}
    if all(v == 0.0 for v in loss.values()):
        return None
    return mesh.MaterialCF(loss, default=0.0)


def material_complex_cfs(mesh, materials):
    """Complex relative permittivity ``eps' - j eps''`` as a CoefficientFunction.

    This is the mass-form coefficient of the **lossy** eigenproblem, which makes B
    complex symmetric (not Hermitian) and the eigenvalue complex. Returns ``None``
    only for an all-vacuum, lossless mapping.
    """
    props = parse_materials(materials)
    if not props:
        return None
    _check_material_names(mesh, props)
    eps = {name: complex(re_, -im) for name, (re_, im) in props.items()}
    if all(v == 1.0 + 0j for v in eps.values()):
        return None
    return mesh.MaterialCF(eps, default=1.0 + 0j)


def max_tan_delta(materials):
    """Largest loss tangent in a materials mapping (0.0 if lossless or empty).

    This is the number the automatic lossless-vs-lossy choice is made on, so the
    *worst* region governs: a single lossy absorber ring in an otherwise low-loss
    structure is what redistributes the mode.
    """
    props = parse_materials(materials)
    return max((im / re_ for re_, im in props.values()), default=0.0)


def resolve_loss_model(materials, eigenmode_config=None):
    """Which dielectric-loss treatment this run uses: ``'lossless'``,
    ``'perturbation'`` or ``'lossy'``.

    Three treatments, one knob:

    - ``'lossless'`` — real eigenproblem, no dielectric loss anywhere. What every
      vacuum cavity has always done, bit for bit.
    - ``'perturbation'`` — the same real eigenproblem, with the dielectric loss
      added afterwards as a volume integral over the lossless field (see
      :meth:`NGSolveMEVP.evaluate_qois`). The error is O(tan_delta^2) in the
      eigenvalue and it cannot capture field redistribution, so it is the right
      tool up to ``tan_delta ~ 1e-2``.
    - ``'lossy'`` — the full complex eigenproblem (complex spaces, complex
      symmetric mass matrix, shift-and-invert Arnoldi). Q comes straight out of
      the complex eigenvalue and the mode *shape* is the lossy one.

    ``eigenmode_config['loss_model']`` selects explicitly; the default (``None`` or
    ``'auto'``) inspects the loss tangents and picks ``'lossy'`` above
    :data:`LOSSY_TAN_DELTA`. Asking for the cheap path anyway is honoured — with a
    warning, because at ``tan_delta ~ 0.1`` a perturbative Q is wrong by percent
    and the frequency is wrong too.
    """
    td = max_tan_delta(materials)
    requested = (eigenmode_config or {}).get('loss_model', None)
    if requested is None:
        requested = 'auto'
    requested = str(requested).lower()
    if requested not in ('auto', 'lossless', 'perturbation', 'lossy'):
        raise ValueError(
            f"eigenmode_config['loss_model']={requested!r} is not one of "
            "'auto' (default), 'lossless', 'perturbation' or 'lossy'.")

    if td == 0.0:
        if requested == 'lossy':
            warnings.warn(
                "eigenmode_config['loss_model']='lossy' but no material has a loss "
                "tangent, so there is no dielectric loss to solve for; using the real "
                "(lossless) eigenproblem. Set tan_delta on a region — via "
                "cav.add_dielectric(..., tan_delta=...) or "
                "eigenmode_config['materials'] — to make the lossy path meaningful.",
                UserWarning, stacklevel=3)
        return 'lossless'

    if requested == 'lossy':
        return 'lossy'
    if requested == 'auto':
        return 'lossy' if td > LOSSY_TAN_DELTA else 'perturbation'

    # Explicit 'lossless'/'perturbation' with real loss present: honour it, but say
    # what it costs. The loss is still reported (perturbatively) — dropping it
    # silently is the one outcome that reads as a correct answer and is not.
    if td > LOSSY_TAN_DELTA:
        warnings.warn(
            f"eigenmode_config['loss_model']={requested!r} was requested but the largest "
            f"loss tangent is {td:.3g} (> {LOSSY_TAN_DELTA:g}), where perturbation theory "
            "is no longer reliable: the O(tan_delta^2) eigenvalue error and the "
            "un-modelled field redistribution both matter. Continuing as asked — Q is "
            "tagged 'perturbation' in the QOIs. Use loss_model='lossy' (or 'auto') for "
            "the full complex eigenproblem.",
            UserWarning, stacklevel=3)
    return 'perturbation'


def surface_resistance(w, conductivity=SIGMA_COPPER, rs=None):
    """Surface resistance [Ohm] at angular frequency *w*.

    Normal conductor: Rs = sqrt(mu0 * w / (2 * conductivity)). For a
    superconductor (or any fixed-Rs material) pass ``rs`` to use that value
    directly instead."""
    if rs is not None:
        return rs
    return np.sqrt(mu0 * w / (2 * conductivity))


@functools.lru_cache(maxsize=None)
def direct_solver_available(name, complex_matrix=False):
    """Whether this NGSolve build can actually factorise with backend *name*.

    Which sparse direct solvers are compiled in varies by platform and by how
    NGSolve was built (pip wheel, conda, source), so this probes a 2x2 problem
    rather than guessing. Cached: the probe runs at most once per backend.

    *complex_matrix* probes a **complex symmetric, indefinite** matrix — the shape
    the lossy eigenproblem factorises in shift-and-invert. It is a genuinely
    different question from the real one: a Cholesky-type backend can be present
    and still refuse it, and finding that out at the eigensolve is a crash deep
    inside Arnoldi rather than a solver-choice message here.
    """
    try:
        face = WorkPlane().Rectangle(1, 1).Face()
        mesh = Mesh(OCCGeometry(face, dim=2).GenerateMesh(maxh=1.0))
        fes = H1(mesh, order=1, complex=complex_matrix)
        u, v = fes.TnT()
        form = grad(u) * grad(v) * dx + u * v * dx
        if complex_matrix:
            # Complex symmetric and indefinite, like (A - shift*B) with a shift
            # inside the spectrum.
            form = grad(u) * grad(v) * dx - (1 + 1j) * u * v * dx
        a = BilinearForm(form).Assemble()
        a.mat.Inverse(fes.FreeDofs(), inverse=name)
        return True
    except Exception:
        return False


def default_direct_solver(complex_matrix=False):
    """Name of the sparse direct-solver backend for the monopole eigenproblem.

    Preference order by platform, filtered by what the build actually provides:

    - Windows: ``pardiso`` (Intel MKL, shipped with the Windows NGSolve build).
    - macOS / Linux: ``umfpack`` (SuiteSparse) first — it is the fast option on
      those platforms, where PARDISO usually is not compiled in.

    Falls back to ``sparsecholesky``, which is always built in. Override per run
    with ``eigenmode_config['direct_solver']``.

    Pass *complex_matrix* for the lossy path, whose shifted matrix is complex
    symmetric and indefinite; a backend that cannot factorise that is skipped
    here rather than failing inside Arnoldi.
    """
    if platform.system() == 'Windows':
        preferred = ('pardiso', 'umfpack')
    else:
        preferred = ('umfpack', 'pardiso')
    for name in preferred:
        if direct_solver_available(name, complex_matrix):
            return name
    if complex_matrix and not direct_solver_available('sparsecholesky', True):
        raise RuntimeError(
            "the lossy (complex) eigenproblem needs a sparse direct solver that can "
            "factorise a complex symmetric indefinite matrix, and this NGSolve build "
            f"provides none of {preferred + ('sparsecholesky',)}. Use "
            "eigenmode_config['loss_model']='lossless' for the perturbative Q, or "
            "install an NGSolve build with UMFPACK or PARDISO.")
    return 'sparsecholesky'


def parse_polarisations(value):
    """Normalize an eigenmode_config 'polarisation' entry to a sorted list of
    azimuthal mode numbers. Accepts an int, a name ('dipole'), or a list of
    either; None defaults to the monopole."""
    if value is None:
        return [0]
    if not isinstance(value, (list, tuple, set)):
        value = [value]
    return sorted({pol_number(v) for v in value})


class NGSolveMEVP:
    """NGSolve-based Maxwell Eigenvalue Problem solver for axisymmetric RF cavities."""

    def __init__(self):
        self.geo = None
        self.step = None
        self.mesh = None
        self.fields = None
        self._last_adaptive_history = None
        # Per-mode dielectric Q of the last solve, read off the complex eigenvalue.
        # None on every path but 'lossy', where it cannot be recovered from the
        # returned fields alone.
        self._last_dielectric_q = None

    @staticmethod
    def requested_n_modes(cav=None, eigenmode_config=None, n_modes=None):
        """Number of user-requested eigenmodes.

        Explicit ``n_modes``/``nmodes`` in the config wins. If no explicit
        value is supplied and *cav* is a cavity-like object, default to
        ``cav.n_cells + 2``; otherwise use the general default of 10.
        """
        if n_modes is None and eigenmode_config:
            n_modes = eigenmode_config.get('n_modes', eigenmode_config.get('nmodes', None))

        if n_modes is None and hasattr(cav, 'n_cells'):
            n_modes = int(cav.n_cells) + 2

        if n_modes is None:
            n_modes = DEFAULT_N_MODES

        if isinstance(n_modes, bool) or int(n_modes) <= 0:
            raise ValueError("'n_modes' must be a positive integer.")

        return int(n_modes)

    @staticmethod
    def pinvit_n_modes(requested_n_modes):
        """PINVIT search size: always two more than requested."""
        return int(requested_n_modes) + 2

    @staticmethod
    def resolve_materials(cav, eigenmode_config=None):
        """``{material: {'eps_r': ..., 'tan_delta': ...}}`` for *cav*, or None if it
        is all vacuum.

        The cavity's own :meth:`~cavsim2d.models.base.Cavity.add_dielectric`
        declarations are the source of truth for which regions exist;
        ``eigenmode_config['materials']`` may override the *properties* of an
        already-declared region, so a sweep over eps_r or tan_delta needs no
        geometry change. Naming a region that was never added raises — otherwise a
        typo would solve the vacuum problem and report it as a dielectric result.

        Overrides **merge** onto the declared properties, so
        ``{'quartz': {'tan_delta': 0.05}}`` is a loss sweep on the declared eps_r
        rather than a silent reset of it to 1.

        ``tan_delta`` is omitted from a region that declares none, so a lossless
        cavity produces exactly the mapping (and hence exactly the QOIs) it always
        did.
        """
        mats = {}
        for d in (getattr(cav, 'dielectrics', ()) or ()):
            props = {'eps_r': float(d['eps_r'])}
            if float(d.get('tan_delta', 0.0) or 0.0):
                props['tan_delta'] = float(d['tan_delta'])
            mats[d['material']] = props
        override = (eigenmode_config or {}).get('materials') or {}
        for name, props in override.items():
            if name not in mats:
                raise ValueError(
                    f"eigenmode_config['materials'] names {name!r}, which is not a "
                    f"dielectric region of this cavity (has {sorted(mats) or 'none'}). "
                    f"Declare it first with cav.add_dielectric({name!r}, eps_r, z=..., "
                    "r=...); this key only overrides the properties of an existing region.")
            if isinstance(props, (int, float, complex)) and not isinstance(props, bool):
                props = {'eps_r': props}
            merged = dict(mats[name])
            merged.update(props)
            if isinstance(merged.get('eps_r'), complex):
                # A complex eps_r override supersedes a declared tan_delta rather
                # than colliding with it (parse_materials refuses both at once).
                merged.pop('tan_delta', None)
            if merged.get('tan_delta') == 0:
                merged.pop('tan_delta')
            mats[name] = merged
        return mats or None

    @staticmethod
    def modes_of_interest(cav, m, eigenmode_config=None, n_modes=None):
        """0-based indices of the modes whose QOIs are reported for polarisation *m*.

        ``eigenmode_config['mode_of_interest']`` is **1-based** — mode 1 is the
        lowest of the passband. A polarisation may have **any number** of modes of
        interest, so each entry is an int or a list of ints of any length. The value
        may be a single entry applied to every polarisation, or a dict keyed by
        polarisation (name or azimuthal number)::

            'mode_of_interest': 9                                    # every polarisation
            'mode_of_interest': [1, 2, 3]                            # three, for every one
            'mode_of_interest': {'monopole': [1, 2, 3, 4], 'dipole': 1}

        Order is preserved and duplicates collapse.

        The **first** mode listed is the primary one: its QOIs become
        ``qois.json``. Every mode of interest is also written to ``qois_moi.json``,
        keyed by its 1-based index.

        Defaults, when the key is absent or a polarisation is missing from the
        dict:

        - monopole (m=0): ``n_cells`` — the pi-mode, the operating mode of an
          accelerating structure. For a 1-cell cavity, gun or pillbox this is
          mode 1, so the convention degrades gracefully to non-accelerator
          geometries.
        - m-pole (m>=1): mode 1, the lowest of the deflecting passband.
        """
        n_cells = int(getattr(cav, 'n_cells', 1) or 1)
        default = n_cells if int(m) == 0 else 1

        requested = (eigenmode_config or {}).get('mode_of_interest', None)
        if isinstance(requested, dict):
            requested = next((requested[k] for k in (pol_name(m), int(m), str(m))
                              if k in requested), None)
        if requested is None:
            requested = default
        if not isinstance(requested, (list, tuple)):
            requested = [requested]
        if len(requested) == 0:
            raise ValueError(
                f"'mode_of_interest' for polarisation {pol_name(m)!r} is empty; give at "
                f"least one 1-based mode index.")

        out = []
        for mode in requested:
            if isinstance(mode, bool) or not isinstance(mode, (int, np.integer)):
                raise ValueError(
                    f"'mode_of_interest' must be a positive integer (1-based), a list of "
                    f"them, or a dict of either; got {mode!r} for polarisation "
                    f"{pol_name(m)!r}.")
            mode = int(mode)
            if mode < 1:
                raise ValueError(
                    f"'mode_of_interest' is 1-based: mode 1 is the lowest of the passband. "
                    f"Got {mode} for polarisation {pol_name(m)!r}.")
            if n_modes is not None and mode > n_modes:
                raise ValueError(
                    f"'mode_of_interest'={mode} for polarisation {pol_name(m)!r} exceeds the "
                    f"{n_modes} modes solved for. Raise eigenmode_config['n_modes'] to at "
                    f"least {mode}.")
            if mode - 1 not in out:
                out.append(mode - 1)
        return out

    # ──────────────────────────────────────────────────────────────────────
    # Geometry
    # ──────────────────────────────────────────────────────────────────────

    def write_geometry_multicell(self, folder, n_cells, multicell,
                                 beampipe='none', plot=False, cell_parameterisation='normal'):
        """Write a multicell cavity geometry file to *folder*."""
        if not os.path.exists(folder):
            try:
                os.mkdir(folder)
            except FileNotFoundError:
                error("Could not create multicell simulation directory. Check folder path.")
                exit()

        file_path = os.path.join(folder, 'geodata.n')
        if cell_parameterisation == 'normal':
            write_cavity_geometry_cli_multicell(n_cells, multicell, beampipe, write=file_path, plot=plot)
        else:
            write_cavity_geometry_cli_flattop(multicell, 'both', n_cell=n_cells, write=file_path)

    def load_geo(self, filepath, output_filepath=None, maxh=1):
        """Load a .geo file via Gmsh, mesh it, and return an NGSolve-ready mesh."""
        if output_filepath is None:
            output_filepath = os.path.dirname(filepath)

        gmsh.initialize()
        gmsh.option.setNumber("General.Verbosity", 0)
        gmsh.option.setNumber("General.Terminal", 0)

        gmsh.open(filepath)
        gmsh.model.mesh.generate(2)
        with suppress_c_stdout_stderr():
            gmsh.write(os.path.join(output_filepath, "mesh.step"))

        self.step_geo = OCCGeometry(os.path.join(output_filepath, "mesh.step"), dim=2)
        self.ngmesh = self.step_geo.GenerateMesh(maxh=maxh)
        self.bcs = self._get_boundaries_from_gmsh()
        gmsh.finalize()

        return self.step_geo, self.ngmesh, self.bcs

    def _build_mesh(self, cav, maxh, order):
        """Return a boundary-tagged, curved NGSolve mesh for *cav*.

        Two backends behind one call:
        - Native: if the cavity exposes a unified ``profile()`` (a geometry
          :class:`~cavsim2d.geometry.Profile`), mesh it directly with
          netgen.occ — exact edges, no gmsh, no ``.geo`` round-trip.
        - Import: otherwise mesh the cavity's ``.geo`` file via gmsh
          (elliptical, spline, imported CAD).
        """
        dielectrics = list(getattr(cav, 'dielectrics', ()) or ())

        maker = getattr(cav, 'profile', None)
        profile = maker() if callable(maker) else None
        if profile is not None:
            region_maxh = {}
            for d in dielectrics:
                # Model API is in mm (like every other cavity dimension); Profile
                # works in metres.
                profile.add_region(d['material'],
                                   z=tuple(v * 1e-3 for v in d['z']),
                                   r=tuple(v * 1e-3 for v in d['r']),
                                   color=d.get('color', (1.0, 1.0, 0.0)))
                if d.get('maxh'):
                    region_maxh[d['material']] = float(d['maxh']) * 1e-3
            return profile.mesh(maxh=maxh, order=order,
                                region_maxh=region_maxh or None)

        if dielectrics:
            raise RuntimeError(
                f"{type(cav).__name__} {cav.name!r} has dielectric regions "
                f"({[d['material'] for d in dielectrics]}) but no native profile(), so it "
                "would be meshed from its .geo file — a path that cannot carry them. gmsh "
                "writes a single Physical Surface and the STEP round-trip drops surface "
                "names entirely, so the regions would vanish and the vacuum problem would "
                "be solved silently. Dielectrics are supported on native-profile cavities "
                "only.")

        if not cav.geo_filepath:
            raise RuntimeError(
                f"{type(cav).__name__} {cav.name!r} has no .geo file and its profile() "
                "returned None, so there is no geometry to mesh. A cavity with "
                "independently varying cells (set_half_cells) is native-only: the gmsh "
                "writer can express uniform mid-cells only, so falling back to it would "
                "silently solve a different cavity.")

        step_geo, ngmesh, bcs = self.load_geo(cav.geo_filepath, maxh=maxh)
        for key, bc in bcs.items():
            ngmesh.SetBCName(key - 1, bc)
        mesh = Mesh(ngmesh)
        mesh.Curve(order)
        return mesh

    def _get_boundaries_from_gmsh(self):
        """Extract boundary condition name map from the current Gmsh model."""
        line_bc_map = {}
        for dim, phys_tag in gmsh.model.getPhysicalGroups():
            if dim != 1:
                continue
            name = gmsh.model.getPhysicalName(dim, phys_tag)
            for line_id in gmsh.model.getEntitiesForPhysicalGroup(dim, phys_tag):
                if isinstance(line_id, tuple):
                    _, line_id = line_id
                line_bc_map[line_id] = name
        return line_bc_map

    # ──────────────────────────────────────────────────────────────────────
    # Solve
    # ──────────────────────────────────────────────────────────────────────

    def solve(self, cav, eigenmode_config=None):
        """Run eigenmode analysis on a single cavity object.

        ``eigenmode_config['polarisation']`` selects the azimuthal mode
        number(s) to solve for: an int, a name ('monopole', 'dipole',
        'quadrupole', 'sextupole', ...), or a list of either. The monopole
        (m=0, default) keeps its flat ``eigenmode/`` layout; every m >= 1
        is solved with the HCurl x H1 product-space formulation and saved
        to ``eigenmode/<pol name>/``.
        """
        eigenmode_folder_structure = {
            'eigenmode': None
        }
        make_dirs_from_dict(eigenmode_folder_structure, cav.self_dir)

        # mesh_config['h'] is in mm, the solver works in metres. The default has to be
        # converted too: leaving it at 20 means a 20 m maxh, i.e. no size constraint at
        # all, so each meshing backend silently picks its own element size.
        mesh_config = (eigenmode_config or {}).get('mesh_config', {})
        mesh_h = mesh_h_metres(mesh_config)
        mesh_p = mesh_config.get('p', 3)
        # Opt-in adaptive (error-driven) h-refinement — applied to EVERY
        # requested polarisation (the recovery-error estimator is m-agnostic).
        adaptive = self._parse_adaptive(mesh_config)
        # Preconditioner for the eigen-solve. Default 'direct'; 'bddc' is ~1.4-1.9x faster
        # on tune-sized monopole solves (same eigenvalue) but is an approximate
        # preconditioner -- unsafe with adaptive refinement (goes stale as the mesh
        # changes) and can stall when many modes share a mesh -- so fall back there.
        self._pre_kind = (eigenmode_config or {}).get('preconditioner', 'direct')
        if self._pre_kind == 'bddc' and adaptive:
            self._pre_kind = 'direct'

        pols = parse_polarisations((eigenmode_config or {}).get('polarisation', 0))

        # One solve path for every polarisation, each on its OWN mesh: with
        # adaptive on, m >= 1 refines to resolve its own deflecting field just
        # as the monopole does, so the meshes (and per-mode convergence) are
        # independent per polarisation.
        for m in sorted(pols):
            self._solve_pol(cav, m, mesh_h, mesh_p, eigenmode_config, adaptive=adaptive)
        return True

    def _solve_pol(self, cav, m, mesh_h, mesh_p, eigenmode_config=None, adaptive=None):
        """Build a mesh for *cav* and solve azimuthal order *m* on it, writing
        all results to ``<cavity>/eigenmode/<pol name>/``.

        ONE path for every polarisation: the eigenproblem (:meth:`_build_system`),
        the recovery-error estimator (:meth:`_error_fields`) and the QOIs
        (:meth:`evaluate_qois`) are all m-agnostic, so this — including the
        adaptive h-refinement — runs identically for the monopole and every
        m-pole. Each polarisation gets its OWN mesh (so its refinement is driven
        by its own field error), built native (``profile()``) or from the
        ``.geo`` file.
        """
        pol_dir = os.path.join(cav.self_dir, 'eigenmode', pol_name(m))
        os.makedirs(pol_dir, exist_ok=True)

        n_modes = self.requested_n_modes(cav, eigenmode_config)
        conductivity = (eigenmode_config or {}).get('conductivity', SIGMA_COPPER)
        rs_ohm = (eigenmode_config or {}).get('surface_resistance', None)
        materials = self.resolve_materials(cav, eigenmode_config)
        # Lossless / perturbative / full complex — decided once, from the loss
        # tangents and eigenmode_config['loss_model'], and reported in the QOIs so a
        # Q is never ambiguous about how it was obtained.
        loss_model = resolve_loss_model(materials, eigenmode_config)

        # Active-length normalisation. Elliptical cavities store the half-cell
        # length as 'L_m'; otherwise take an explicit 'normalization_length'
        # (monopole) and finally let evaluate_qois fall back to the on-axis
        # field extent (L=None).
        L_norm = cav.parameters.get('L_m', None)
        if L_norm is None:
            L_norm = (eigenmode_config or {}).get('normalization_length', None) if m == 0 else 1

        # Solve on this polarisation's own mesh; adaptive refines it in place to
        # resolve *this* polarisation's modes (adaptive=None -> single solve).
        mesh = self._build_mesh(cav, mesh_h, mesh_p)
        freq_fes, gfu_E, gfu_H = self._solve_eigenproblem(cav, pol_dir, mesh, mesh_p,
                                                          n_modes, m=m, adaptive=adaptive,
                                                          materials=materials,
                                                          loss_model=loss_model)
        q_diel = self._last_dielectric_q

        # Frequency-only fast path (tuning inner solves). A root-finder needs only
        # the mode-of-interest frequency, which is the eigenvalue itself (freq_fes is
        # already MHz). Skip the QOI field integrals (evaluate_qois runs once per mode
        # of interest AND once per mode) and the mesh/field disk writes; emit a minimal
        # qois.json so the frequency read (pyTuner -> monopole/qois.json['freq [MHz]'],
        # qois_df) is unchanged. The primary mode is exactly the one the full path
        # reports, so the tuned frequency is identical -- only the QOIs are deferred to
        # the final full solve once tuning is complete.
        if (eigenmode_config or {}).get('freq_only'):
            moi = self.modes_of_interest(cav, m, eigenmode_config, len(freq_fes))
            primary = min((moi[0] if moi else 0), len(freq_fes) - 1)
            q = {'freq [MHz]': float(freq_fes[primary]), 'm': int(m),
                 'polarisation': pol_name(m), 'mode_of_interest': str(primary + 1)}
            with open(os.path.join(pol_dir, 'qois.json'), 'w') as f:
                json.dump(q, f, indent=4, separators=(',', ': '))
            with open(os.path.join(pol_dir, 'qois_moi.json'), 'w') as f:
                json.dump({str(primary + 1): q}, f, indent=4, separators=(',', ': '))
            # Minimal all-modes file (primary mode only) so ``qois_df`` stays valid.
            with open(os.path.join(pol_dir, 'qois_all_modes.json'), 'w') as f:
                json.dump({str(primary): q}, f, indent=4, separators=(',', ': '))
            return True

        # Save after solving: adaptive refinement mutates *mesh* in place, so
        # this persists the finest mesh actually used for the QOIs.
        self.save_mesh(pol_dir, mesh)

        # Modes of interest: the pi-mode for the monopole passband, mode 1 (the
        # lowest of the deflecting passband) for m >= 1 — see
        # :meth:`modes_of_interest`. The first listed is primary (-> qois.json).
        moi = self.modes_of_interest(cav, m, eigenmode_config, len(freq_fes))
        if m == 0:
            n_dofs = HCurl(mesh, order=mesh_p, dirichlet="PEC").ndof
        else:
            fes_rz = HCurl(mesh, order=mesh_p, dirichlet="PEC")
            _, fes_phi = fes_rz.CreateGradient()
            n_dofs = fes_rz.ndof + fes_phi.ndof

        qois_moi = {}
        for idx in moi:
            q = self.evaluate_qois(mesh, gfu_E, gfu_H, freq_fes, m, mode_idx=idx,
                                   n_cells=cav.n_cells, L=L_norm, save_dir=pol_dir,
                                   conductivity=conductivity, surface_resistance_ohm=rs_ohm,
                                   write_axis=(idx == moi[0]), materials=materials,
                                   loss_model=loss_model, q_diel=q_diel)
            # String metadata (UQ coerces the qois table to numerics and drops
            # text columns; a bare int would be averaged into nonsense stats).
            q['mode_of_interest'] = str(idx + 1)
            q["No of DOFs"] = n_dofs
            qois_moi[str(idx + 1)] = q

        with open(os.path.join(pol_dir, 'qois_moi.json'), "w") as f:
            json.dump(qois_moi, f, indent=4, separators=(',', ': '))
        # qois.json is the primary (first) mode of interest.
        with open(os.path.join(pol_dir, 'qois.json'), "w") as f:
            json.dump(qois_moi[str(moi[0] + 1)], f, indent=4, separators=(',', ': '))

        qois_all_modes = {}
        for ii in range(len(freq_fes)):
            qois_all_modes[ii] = self.evaluate_qois(mesh, gfu_E, gfu_H, freq_fes, m, mode_idx=ii,
                                                    n_cells=cav.n_cells, L=L_norm, save_dir=pol_dir,
                                                    conductivity=conductivity, surface_resistance_ohm=rs_ohm,
                                                    materials=materials, loss_model=loss_model,
                                                    q_diel=q_diel)

        with open(os.path.join(pol_dir, 'qois_all_modes.json'), "w") as f:
            json.dump(qois_all_modes, f, indent=4, separators=(',', ': '))

        return True

    def cavity_multicell(self, no_of_cells=1, no_of_modules=1, multicell=None,
                         fid=None, bc=33, pol='monopole', f_shift='default', beta=1,
                         n_modes=None, beampipes='None', sim_folder='NGSolveMEVP',
                         parentDir=None, projectDir=None, subdir='',
                         expansion=None, expansion_r=None, mesh_args=None, opt=False,
                         deformation_params=None, eigenmode_config=None):
        """Write geometry and run eigenmode analysis for a multicell cavity."""
        if mesh_args is None:
            mesh_args = [20, 20]

        # Unified eigenmode folder — flat structure, no Cavities/ subfolder
        if opt:
            run_save_directory = projectDir / f'{fid}/eigenmode'
        elif subdir == '':
            run_save_directory = projectDir / f'{fid}/eigenmode'
        else:
            run_save_directory = projectDir / f'{subdir}/eigenmode/{fid}'

        self.write_geometry_multicell(run_save_directory, no_of_cells, multicell,
                                      beampipes, plot=False)

        if not os.path.exists(os.path.join(run_save_directory, 'geodata.geo')):
            error('Could not run eigenmode analysis due to error in geometry.')
            return False

        mesh_h, mesh_p = 20, 3
        A_m, B_m, a_m, b_m, Ri_m, L, Req = np.array(multicell[:7])
        maxh = L / mesh_h * 1e-3

        gmsh.initialize()
        gmsh.option.setNumber("General.Verbosity", 0)
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.open(os.path.join(run_save_directory, "geodata.geo"))
        gmsh.model.mesh.generate(2)

        with suppress_c_stdout_stderr():
            gmsh.write(os.path.join(run_save_directory, "mesh.step"))

        step_geo = OCCGeometry(os.path.join(run_save_directory, "mesh.step"), dim=2)
        ngmesh = step_geo.GenerateMesh(maxh=maxh)
        bcs = self._get_boundaries_from_gmsh()
        gmsh.finalize()

        for key, bc_name in bcs.items():
            ngmesh.SetBCName(key - 1, bc_name)

        mesh = Mesh(ngmesh)
        mesh.Curve(mesh_p)
        self.save_mesh(run_save_directory, mesh)

        n_modes = self.requested_n_modes(n_modes=n_modes if n_modes is not None else no_of_cells + 2)
        freq_fes, gfu_E, gfu_H = self._solve_eigenproblem(run_save_directory, run_save_directory, mesh, mesh_p, n_modes)

        qois = self.evaluate_qois(mesh, gfu_E, gfu_H, freq_fes, mode_idx=no_of_cells - 1, n_cells=no_of_cells,
                                  L=L, save_dir=run_save_directory, write_axis=True)

        with open(os.path.join(run_save_directory, 'qois.json'), "w") as f:
            json.dump(qois, f, indent=4, separators=(',', ': '))

        qois_all_modes = {}
        for ii in range(len(freq_fes)):
            qois_all_modes[ii] = self.evaluate_qois(mesh, gfu_E, gfu_H, freq_fes, mode_idx=ii, n_cells=no_of_cells, L=L)

        with open(os.path.join(run_save_directory, 'qois_all_modes.json'), "w") as f:
            json.dump(qois_all_modes, f, indent=4, separators=(',', ': '))

        return True

    def cavity_quarter(self, cell, bp=False, save_dir=None, bc=33, n_modes=1,
                       mesh_h=12, mesh_p=3, m=0, mode_index=0):
        r"""Eigenfrequency of a single **quarter cell** — a half-cell with a
        PMC boundary on the equator mid-plane — used to tune each half-cell's length
        ``L`` to the target frequency (the Corno et al. / WEPB015 per-half-cell
        method). ``cell`` is the half-cell parameter array
        ``[A, B, a, b, Ri, L, Req, (l)]`` in **mm**; ``bp=True`` attaches a beampipe
        (an **end** cup), ``bp=False`` is a **mid** cup with no beampipe.

        ``m`` selects the azimuthal order (0 monopole, 1 dipole, ...) and
        ``mode_index`` the mode within that band (0 = the fundamental of the band).
        The default ``m=0, mode_index=0`` is the accelerating fundamental — the
        quarter-cell resonance that maps to the assembled cavity's pi-mode.

        The quarter cell is a small cavity in its own right, so ``save_dir`` is laid
        out like a normal cavity: the ``.geo`` + ``mesh.step`` go in
        ``save_dir/geometry/``, the solved fields + ``qois.json`` in
        ``save_dir/eigenmode/`` — a caller (the tuner) adds a ``tune/`` sibling for
        the tuning record. ``mesh_h=12`` (a quarter cell is over-resolved at 20 —
        the frequency is identical to 3 significant figures but ~30 % faster).

        Returns the selected mode's frequency [MHz], or ``None`` on degenerate
        geometry. Mirrors :meth:`cavity_multicell` but writes the quarter geometry
        via the model's :meth:`EllipticalCavity.write_quarter_geometry`."""
        # Deferred: the geometry writer lives on the model (models <-> solvers cycle).
        from cavsim2d.models.elliptical import EllipticalCavity
        names = ['A', 'B', 'a', 'b', 'Ri', 'L', 'Req']
        bp_str = 'left' if bp else 'none'
        suffix = '_el' if bp else '_m'
        params = {f'{nm}{suffix}': float(cell[i]) for i, nm in enumerate(names)}

        geo_dir = os.path.join(save_dir, 'geometry')
        eigen_dir = os.path.join(save_dir, 'eigenmode')
        os.makedirs(geo_dir, exist_ok=True)
        os.makedirs(eigen_dir, exist_ok=True)
        geo_stub = os.path.join(geo_dir, 'geodata.n')
        # write_quarter_geometry does not touch instance state, so call it unbound.
        EllipticalCavity.write_quarter_geometry(None, params, bp=bp_str,
                                                write=geo_stub, ignore_degenerate=True)
        geo_path = geo_stub.replace('.n', '.geo')
        if not os.path.exists(geo_path):
            return None

        L = float(cell[5])
        maxh = L / mesh_h * 1e-3

        gmsh.initialize()
        gmsh.option.setNumber("General.Verbosity", 0)
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.open(geo_path)
        gmsh.model.mesh.generate(2)
        with suppress_c_stdout_stderr():
            gmsh.write(os.path.join(geo_dir, "mesh.step"))
        step_geo = OCCGeometry(os.path.join(geo_dir, "mesh.step"), dim=2)
        ngmesh = step_geo.GenerateMesh(maxh=maxh)
        bcs = self._get_boundaries_from_gmsh()
        gmsh.finalize()
        for key, bc_name in bcs.items():
            ngmesh.SetBCName(key - 1, bc_name)

        mesh = Mesh(ngmesh)
        mesh.Curve(mesh_p)
        self.save_mesh(geo_dir, mesh)

        want = self.requested_n_modes(n_modes=max(int(n_modes), int(mode_index) + 1))
        freq_fes, gfu_E, gfu_H = self._solve_eigenproblem(
            eigen_dir, eigen_dir, mesh, mesh_p, want, m=int(m))
        if freq_fes is None or len(freq_fes) == 0:
            return None
        idx = min(int(mode_index), len(freq_fes) - 1)
        qois = self.evaluate_qois(mesh, gfu_E, gfu_H, freq_fes, m=int(m),
                                  mode_idx=idx, n_cells=1, L=L, save_dir=eigen_dir)
        with open(os.path.join(eigen_dir, 'qois.json'), 'w') as f:
            json.dump(qois, f, indent=4, separators=(',', ': '))
        return float(qois['freq [MHz]'])

    @staticmethod
    def _parse_adaptive(mesh_config):
        """Normalise ``mesh_config['adaptive']`` to a settings dict, or None if off.

        Adaptive mesh refinement is opt-in: pass ``True`` to enable it with the
        defaults, or a dict to override them::

            'adaptive': True
            'adaptive': {'tol': 1e-12, 'max_refinements': 8, 'max_ndof': 100000,
                         'theta': 0.25}

        - ``tol``: error tolerance. Refinement stops once **every** solved
          mode's recovery ``max_err`` has fallen below this (default 1e-12).
        - ``theta``: Doerfler marking fraction — elements with error above
          ``theta * max_error`` are refined.
        - ``max_ndof`` / ``max_refinements``: hard caps; stop once either is
          reached even if the tolerance is not met.
        """
        a = (mesh_config or {}).get('adaptive', None)
        if a is None or a is False:
            return None
        cfg = dict(a) if isinstance(a, dict) else {}
        cfg.setdefault('tol', 1e-12)
        cfg.setdefault('theta', 0.25)
        cfg.setdefault('max_ndof', 100000)
        cfg.setdefault('max_refinements', 8)
        return cfg

    def _build_system(self, mesh, mesh_p, m_pol, f_shift=0, direct_solver=None,
                      materials=None, complex_fes=False):
        """Build the reusable space, forms and preconditioner for azimuthal order
        *m_pol* — the single formulation used for **every** polarisation.

        The field ansatz is E = (E_r, E_z) cos(m phi) + E_phi sin(m phi) e_phi
        with the scaled azimuthal unknown ``u_phi = r * E_phi``, discretised on
        the product space ``HCurl x H1``. Two details make it valid for all m:

        - ``fes_phi`` is an H1 space of order ``mesh_p + 1`` that is **zero on
          the axis** (``dirichlet="PEC|AXI"``), not the space returned by
          ``CreateGradient``. The CreateGradient space (same ndof, PEC-only
          dirichlet) is used solely for the gradient-kernel projector.
        - the kernel (u, u_phi) = (grad psi, m psi) is removed with a
          b-orthogonal projector built from ``G = embU @ Grz - m * embPhi``.

        At **m = 0** the two blocks decouple: the HCurl block gives the monopole
        TM modes and the H1 block the monopole **TE** modes — which an HCurl-only
        formulation cannot represent at all. Validated against the analytical
        pillbox spectrum (TM and TE) for m = 0..8.

        The space and forms are created **once** and reused across adaptive
        refinements (each pass calls ``fes.Update()`` and reassembles). Creating
        a fresh space per pass instead leaves stale spaces registered on the
        mesh, and netgen's ``Refine()`` then updates them onto freed memory —
        an access violation on the second refinement.

        With *complex_fes* the same forms are assembled in **complex** arithmetic and
        the mass coefficient becomes ``eps' - j eps''``. That makes B complex
        symmetric rather than Hermitian, so the eigenvalue is complex and the
        Hermitian eigensolver (PINVIT) no longer applies — this system is solved by
        :meth:`_solve_lossy_system` instead of :meth:`_solve_system`.
        """
        if direct_solver is None:
            direct_solver = default_direct_solver(complex_fes)
        # p >= 2 is required: at p=1 the HCurl(p) x H1(p+1) product space is
        # rank-deficient for this formulation and the PINVIT reduced eigenproblem
        # comes out with NaNs (scipy.linalg.eigh then raises a cryptic "array must
        # not contain infs or NaNs"). Fail early with a clear message instead.
        if int(mesh_p) < 2:
            raise ValueError(
                f"eigenmode polynomial order p={mesh_p} is unsupported: the "
                "HCurl(p) x H1(p+1) formulation is rank-deficient at p<2 and the "
                "eigensolve returns NaN. Use p>=2 (set via mesh_config['p'], "
                "default 3).")
        r = y
        fes_rz = HCurl(mesh, order=mesh_p, dirichlet="PEC", complex=complex_fes)
        fes_phi = H1(mesh, order=mesh_p + 1, dirichlet="PEC|AXI", complex=complex_fes)
        fes = fes_rz * fes_phi
        (u, u_phi), (v, v_phi) = fes.TnT()

        # Dielectric weighting. Weighting the mass form by eps_r solves
        # curl(curl E) = lambda * eps_r * E, so lambda is STILL k0^2 = (w/c0)^2 and
        # the frequency conversion below is unchanged. eps_r is a single scalar
        # factor on the whole bracket because every term comes from the same
        # |E|^2. eps_cf is None for a vacuum cavity, in which case the plain
        # expression is assembled verbatim.
        #
        # No interface treatment is needed: HCurl enforces tangential-E continuity
        # and u_phi = r*E_phi is tangential to any r-z interface, which are exactly
        # the physical dielectric interface conditions. Normal D continuity is
        # natural. The gradient kernel is also untouched, so the b-orthogonal
        # projector in _solve_system stays valid (curl of a gradient vanishes
        # pointwise, and B stays SPD for eps_r > 0).
        # The lossy path weights the mass form by the COMPLEX permittivity; the
        # lossless/perturbative path keeps the real one, so its assembled matrices
        # are bit-for-bit what they always were.
        eps_cf = (material_complex_cfs(mesh, materials) if complex_fes
                  else material_cfs(mesh, materials))

        stiff = (r * curl(u) * curl(v)
                 + 1 / r * (m_pol**2 * u * v
                            + m_pol * u * grad(v_phi)
                            + m_pol * grad(u_phi) * v
                            + grad(u_phi) * grad(v_phi))) * dx
        mass_expr = (r * u * v + 1 / r * u_phi * v_phi)
        mass = (mass_expr if eps_cf is None else eps_cf * mass_expr) * dx

        # Search around a frequency if a shift is provided. Not on the complex
        # path: shift-and-invert Arnoldi applies its own shift, so folding one into
        # `a` here would apply it twice.
        if f_shift and f_shift != 'default' and not complex_fes:
            shift_lam = (2 * pi * f_shift * 1e6 / c0)**2
            a = BilinearForm(stiff - shift_lam * mass)
        else:
            a = BilinearForm(stiff)
        b = BilinearForm(mass)

        # Preconditioner choice for the eigen-solve's `pre` operator. 'direct' (default)
        # factorises (stiff + mass); 'bddc' registers an iterative BDDC preconditioner
        # (~1.4-1.9x faster on tune-sized monopole solves, same eigenvalue). BDDC must be
        # registered on its form before assembly, so build (stiff + mass) here where the
        # expression lives. BDDC is an APPROXIMATE preconditioner: safe for the tune's
        # single monopole solves, but not for adaptive refinement or many shared-mesh
        # modes, so it stays opt-in (`eigenmode_config['preconditioner'] = 'bddc'`).
        pre_kind = getattr(self, '_pre_kind', 'direct')
        ab_form = pre_reg = None
        if pre_kind == 'bddc':
            ab_form = BilinearForm(stiff + mass)
            pre_reg = Preconditioner(ab_form, "bddc")

        return {'fes': fes, 'fes_rz': fes_rz, 'fes_phi': fes_phi,
                'a': a, 'b': b, 'm': m_pol, 'eps_cf': eps_cf,
                'f_shift': f_shift, 'direct_solver': direct_solver,
                'complex_fes': bool(complex_fes),
                'ab_form': ab_form, 'pre_reg': pre_reg, 'pre_kind': pre_kind}

    def _solve_system(self, system, n_modes, pinvit_maxit=20):
        """Update, assemble and solve the reusable *system* on its (possibly
        just-refined) mesh. Returns ``(freq_fes, gfu_E, gfu_H)`` where each
        ``gfu_E`` entry is a product-space GridFunction (components: in-plane
        (E_z, E_r), u_phi = r*E_phi) and each ``gfu_H`` entry is a pair
        ``(H_inplane_cf, H_phi_cf)`` of azimuthal envelope coefficient
        functions. The representation is the same for every m."""
        fes, fes_rz = system['fes'], system['fes_rz']
        a, b, m_pol = system['a'], system['b'], system['m']
        f_shift, direct_solver = system['f_shift'], system['direct_solver']
        r = y

        with TaskManager():
            fes.Update()
            a.Assemble()
            b.Assemble()

            if system.get('pre_kind') == 'bddc':
                system['ab_form'].Assemble()          # builds the BDDC preconditioner
                pre = system['pre_reg']
            else:
                pre = (a.mat + b.mat).CreateSparseMatrix().Inverse(fes.FreeDofs())

            # Remove the gradient kernel (u, u_phi) = (grad psi, m psi). The
            # potential space comes from CreateGradient (PEC-only dirichlet);
            # it has the same ndof as fes_phi, so the two embeddings combine.
            Grz, fes_pot = fes_rz.CreateGradient()
            embU, embPhi = fes.embeddings
            G = (embU @ Grz - m_pol * embPhi).CreateSparseMatrix()
            GT = G.CreateTranspose()
            invh1 = (GT @ b.mat @ G).Inverse(inverse=direct_solver,
                                             freedofs=fes_pot.FreeDofs())
            projpre = (IdentityMatrix(fes.ndof) - G @ invh1 @ GT @ b.mat) @ pre

            evals_, evecs_ = solvers.PINVIT(a.mat, b.mat, pre=projpre,
                                            num=self.pinvit_n_modes(n_modes),
                                            maxit=pinvit_maxit, printrates=False)
            # Drop any residual gradient-kernel mode. The threshold is RELATIVE:
            # kernel eigenvalues sit at ~1e-10 while physical ones are ~1e3, so
            # this selects identically to the old absolute `> 1` on every vacuum
            # cavity — but `> 1` is a fixed 47.7 MHz floor, and a high-eps_r fill
            # divides lambda by eps_r, which could push a genuine mode under it.
            evals_ = np.array(evals_)
            lam_max = float(np.max(evals_)) if len(evals_) else 0.0
            mask_ = evals_ > 1e-6 * lam_max
            evals = evals_[mask_]
            evecs = np.array(evecs_)[mask_]

            if f_shift and f_shift != 'default':
                shift_lam = (2 * pi * f_shift * 1e6 / c0)**2
                freq_fes = [c0 * np.sqrt(np.abs(lam + shift_lam)) / (2 * np.pi) * 1e-6 for lam in evals]
            else:
                freq_fes = [c0 * np.sqrt(np.abs(lam)) / (2 * np.pi) * 1e-6 for lam in evals]

            # 1/r is singular on the axis; H_inplane there is defined by its
            # limit, so clamp the weight instead of dividing by zero (the axis
            # is a measure-zero set in every integral that uses it).
            inv_r = IfPos(r - AXIS_EPS, 1 / r, 0)

            gfu_E = []
            gfu_H = []
            for i in range(len(evecs)):
                w = 2 * pi * freq_fes[i] * 1e6
                gfu = GridFunction(fes)
                gfu.vec.data = evecs[i]
                gfu_E.append(gfu)

                u_gf, uphi_gf = gfu.components
                # Azimuthal envelopes of H = curl(E) / (-i mu0 w); the phase
                # factor is dropped since only magnitudes enter the QOIs.
                H_inplane = inv_r / (mu0 * w) * (m_pol * u_gf + grad(uphi_gf))
                H_phi = 1 / (mu0 * w) * curl(u_gf)
                gfu_H.append((H_inplane, H_phi))

        return freq_fes, gfu_E, gfu_H

    @staticmethod
    def _dedupe_shifts(lams, rtol=1e-3):
        """Distinct shift-and-invert centres for a list of eigenvalues.

        Nearly-degenerate modes (a dipole pair, the two ends of a flat passband)
        would otherwise each pay for their own factorisation and return the same
        cluster twice. Values within *rtol* of one already kept collapse onto it.
        """
        shifts = []
        for lam in sorted(float(np.real(l)) for l in lams):
            if lam <= 0:
                continue
            if shifts and abs(lam - shifts[-1]) <= rtol * abs(lam):
                continue
            shifts.append(lam)
        return shifts

    @staticmethod
    def _normalise_mode(mesh, gfu, eps_re_cf):
        """Put a lossy eigenvector on the same scale (and phase) as a real one.

        An eigenvector has no intrinsic amplitude, so every *absolute* QOI —
        ``U``, ``Ploss``, ``Pdiel``, ``Epk``, ``Hpk``, ``Vacc``, ``Eacc`` — is
        reported at whatever scale the eigensolver happened to return. PINVIT
        returns B-normalised vectors, so the real path has a fixed convention;
        Arnoldi does not, and its scale is not even stable between two runs of the
        same problem. Left alone, the lossy path reports absolute quantities that
        change run to run and disagree with the lossless path in the small-loss
        limit, where the two must agree.

        The convention imposed here is the real path's, written with the **real**
        permittivity, ``integral eps_r' |E|^2 r dA = 1``. Using eps_r' (not the
        complex eps_r) is what makes the two paths converge as tan_delta -> 0.

        The global phase is fixed too — the largest coefficient is rotated onto the
        positive real axis. No QOI depends on it (they all go through |.| or
        ``abs``), but it makes a saved lossy field reproducible instead of differing
        by an arbitrary phase on every run.
        """
        u_gf, uphi_gf = gfu.components
        density = (y * InnerProduct(u_gf, u_gf)
                   + 1 / y * uphi_gf * Conj(uphi_gf))
        if eps_re_cf is not None:
            density = eps_re_cf * density
        norm2 = Integrate(density, mesh).real
        if norm2 <= 0:
            return
        fv = gfu.vec.FV().NumPy()
        scale = 1.0 / np.sqrt(norm2)
        k = int(np.argmax(np.abs(fv)))
        if fv[k] != 0:
            scale = scale * abs(fv[k]) / fv[k]
        fv *= scale

    @staticmethod
    def _eigen_residual(mat_a, mat_b, lam, vec, free_mask=None):
        """Relative eigenpair residual ``||A x - lam B x|| / (|lam| ||B x||)``.

        The convergence test for a single Arnoldi Ritz pair. Constrained DOFs are
        masked out (*free_mask*, a boolean array over the DOFs): the forms carry no
        equation there, so whatever sits on those rows is not part of the residual.
        """
        ax = mat_a.CreateColVector()
        bx = mat_b.CreateColVector()
        ax.data = mat_a * vec
        bx.data = mat_b * vec
        a_np = np.asarray(ax.FV().NumPy())
        b_np = np.asarray(bx.FV().NumPy())
        if free_mask is not None:
            a_np, b_np = a_np[free_mask], b_np[free_mask]
        denom = abs(lam) * np.linalg.norm(b_np)
        if denom == 0:
            return np.inf
        return float(np.linalg.norm(a_np - lam * b_np) / denom)

    def _solve_lossy_system(self, mesh, mesh_p, m_pol, materials, lam_shifts,
                            n_modes, direct_solver=None, n_arnoldi=None):
        """Solve the **complex** (lossy-dielectric) eigenproblem by shift-and-invert
        Arnoldi, returning ``(freq_fes, gfu_E, gfu_H, q_diel)``.

        With eps_r = eps' - j eps'' the mass matrix is complex **symmetric**, not
        Hermitian: lambda = (w/c0)^2 is complex, so PINVIT/LOBPCG (which assume a
        Hermitian pencil) do not apply and the spectrum has to be reached with
        Arnoldi on ``(A - sigma B)^-1 B``. On a 2D meridian mesh the shifted matrix
        is small enough to factorise directly, so no preconditioner engineering is
        needed.

        **The shifts come from the lossless solve.** That is what makes this
        tractable: the gradient kernel is still there at lambda ~ 0 and it is huge
        (one mode per potential DOF), so a shift far from the band lets the kernel
        cluster — all of it at |1/sigma| under the shift-invert map — crowd the
        Krylov space and starve the physical modes. Centred on a lossless
        eigenvalue the physical mode maps to ~1/(sigma*tan_delta), orders of
        magnitude above the kernel, and Arnoldi finds it first. Each shift is
        pulled 0.1% off its eigenvalue so that a *lossless* material (a user who
        forces ``loss_model='lossy'`` with tan_delta = 0) cannot land on an exactly
        singular matrix.

        Each shift contributes the one eigenpair it converges best, and every pair
        is gated on its residual before it counts as a mode; a shift that falls
        short is re-centred on the eigenvalue it just found and tried again. That
        second pass is what a large tan_delta needs: the lossless shift is only a
        good centre while the mode has not moved far, and by tan_delta ~ 0.1 the
        upper modes come back half-converged from it. The accepted modes are then
        deduplicated (two shifts can land on the same mode once the loss moves the
        spectrum appreciably) and sorted by Re(lambda), which recovers the passband
        in order.

        Q_diel is read straight off the eigenvalue: w = c0*sqrt(lambda) has
        ``|E| ~ exp(-Im(w) t)``, energy ~ exp(-2 Im(w) t) = exp(-Re(w) t / Q), so
        ``Q = Re(w) / (2 Im(w))``. Magnitudes are taken so the answer does not
        depend on which complex-conjugate branch the eigensolver returns.
        """
        if direct_solver is None:
            direct_solver = default_direct_solver(True)
        n_arnoldi = int(n_arnoldi or DEFAULT_ARNOLDI_VECTORS)
        shifts = self._dedupe_shifts(lam_shifts)
        if not shifts:
            raise RuntimeError(
                "the lossy eigensolve has no shifts to centre on — the lossless solve "
                "returned no positive eigenvalue to seed it with.")

        system = self._build_system(mesh, mesh_p, m_pol, 0, direct_solver, materials,
                                    complex_fes=True)
        fes, a, b = system['fes'], system['a'], system['b']

        with TaskManager():
            fes.Update()
            a.Assemble()
            b.Assemble()
            free_mask = np.fromiter((bool(d) for d in fes.FreeDofs()),
                                    dtype=bool, count=fes.ndof)

            def arnoldi_at(sigma):
                """Best (lowest-residual) physical eigenpair near complex *sigma*."""
                vecs = [GridFunction(fes).vec.CreateVector() for _ in range(n_arnoldi)]
                lam = ArnoldiSolver(a.mat, b.mat, fes.FreeDofs(), vecs, complex(sigma),
                                    inverse=direct_solver)
                pairs = [(complex(l), v) for l, v in zip(lam, vecs)]
                # Same relative kernel filter as the real path: the gradient modes
                # sit at ~1e-10 of the physical eigenvalues, and an absolute floor
                # would be a fixed frequency floor that a high-eps_r fill can push a
                # real mode under.
                lam_max = max((abs(l) for l, _ in pairs), default=0.0)
                pairs = [(l, v) for l, v in pairs
                         if abs(l) > 1e-6 * lam_max and l.real > 0]
                if not pairs:
                    return None
                scored = [(self._eigen_residual(a.mat, b.mat, l, v, free_mask), l, v)
                          for l, v in pairs]
                return min(scored, key=lambda t: t[0])

            # One mode per shift, refined until it converges. Arnoldi returns
            # len(vecs) Ritz pairs whether or not they converged, and an unconverged
            # one is the dangerous kind of wrong — it looks like an extra mode at a
            # plausible frequency — so the relative residual
            # ||A x - lam B x|| / (|lam| ||B x||) is the gate. A *lossless* shift is
            # only a good centre while the loss is small: at tan_delta ~ 0.1 the mode
            # has moved further than the shift-invert map tolerates and the high
            # modes come back half-converged. Re-shifting onto the eigenvalue just
            # found fixes that in one more pass, and costs a factorisation only for
            # the modes that actually needed it.
            targets = [complex(sig * (1 - 1e-3)) for sig in shifts]
            accepted, poor = [], []
            for _pass in range(ARNOLDI_MAX_PASSES):
                retry = []
                for sigma in targets:
                    best = arnoldi_at(sigma)
                    if best is None:
                        continue
                    res, lam, vec = best
                    if res < ARNOLDI_RESIDUAL_TOL:
                        accepted.append((lam, vec))
                    else:
                        # Shift onto the eigenvalue itself, held 1e-4 off so the
                        # shifted matrix cannot be exactly singular.
                        retry.append((res, lam, vec, lam * (1 - 1e-4)))
                poor = retry
                targets = [t[3] for t in retry]
                if not targets:
                    break
            for res, lam, vec, _ in poor:
                # Still short of tolerance after the last pass. Dropping a real mode
                # would renumber every mode above it, which is worse than reporting a
                # slightly loose one — so the bar for keeping it is generous, and only
                # a residual big enough to mean "not a mode" drops it.
                if res < ARNOLDI_RESIDUAL_WARN:
                    accepted.append((lam, vec))
                elif res < ARNOLDI_RESIDUAL_ACCEPT:
                    # warnings.warn, not the verbosity-gated warning(): a mode that
                    # did not converge is a correctness problem, and a silent one
                    # reads as a clean result.
                    warnings.warn(
                        f"lossy eigensolve: the mode near "
                        f"{c0 * np.sqrt(lam).real / (2 * np.pi) * 1e-6:.3f} MHz converged "
                        f"only to a relative residual of {res:.1e}. Raise "
                        f"eigenmode_config['arnoldi_vectors'] (currently {n_arnoldi}) "
                        f"if that accuracy is not enough.", UserWarning, stacklevel=2)
                    accepted.append((lam, vec))
                else:
                    warnings.warn(
                        f"lossy eigensolve: dropping an unconverged eigenvalue "
                        f"(relative residual {res:.1e}) rather than reporting it as a "
                        f"mode. The remaining modes are renumbered, so check "
                        f"'mode_of_interest' against the reported frequencies.",
                        UserWarning, stacklevel=2)

            accepted.sort(key=lambda t: t[0].real)
            modes = []
            for lam, vec in accepted:
                # Distinct shifts can converge onto the same mode once the loss is
                # large enough to move the modes appreciably. The tolerance is loose
                # because two Arnoldi runs converge the same mode to slightly
                # different residuals, not to the same digits.
                if modes and abs(lam - modes[-1][0]) <= 1e-6 * abs(lam):
                    continue
                modes.append((lam, vec))
            if len(modes) < len(shifts):
                warnings.warn(
                    f"lossy eigensolve: {len(modes)} distinct mode(s) for "
                    f"{len(shifts)} requested — two shifts converged onto the same "
                    f"mode. The modes are renumbered, so check 'mode_of_interest' "
                    f"against the reported frequencies; raising "
                    f"eigenmode_config['arnoldi_vectors'] (currently {n_arnoldi}) may "
                    f"separate them.", UserWarning, stacklevel=2)

            inv_r = IfPos(y - AXIS_EPS, 1 / y, 0)
            eps_re_cf = material_cfs(mesh, materials)
            freq_fes, gfu_E, gfu_H, q_diel = [], [], [], []
            for lam, vec in modes:
                w_c = c0 * np.sqrt(complex(lam))          # complex angular frequency
                freq = abs(w_c.real) / (2 * np.pi) * 1e-6
                freq_fes.append(freq)
                q_diel.append(abs(w_c.real) / (2 * abs(w_c.imag)) if w_c.imag else np.inf)

                gfu = GridFunction(fes)
                gfu.vec.data = vec
                self._normalise_mode(mesh, gfu, eps_re_cf)
                gfu_E.append(gfu)
                u_gf, uphi_gf = gfu.components
                w = 2 * pi * freq * 1e6
                gfu_H.append((inv_r / (mu0 * w) * (m_pol * u_gf + grad(uphi_gf)),
                              1 / (mu0 * w) * curl(u_gf)))

        return freq_fes, gfu_E, gfu_H, q_diel

    def _solve_modes(self, mesh, mesh_p, m_pol, n_modes, save_dir=None,
                     f_shift=0, direct_solver=None, pinvit_maxit=20, materials=None,
                     loss_model='lossless', n_arnoldi=None):
        """Solve the Maxwell eigenproblem for a single azimuthal order *m_pol*.

        The one entry point for every polarisation (m = 0 monopole included);
        builds a fresh system and solves it. See :meth:`_build_system`.

        *loss_model* (see :func:`resolve_loss_model`) selects the treatment of a
        complex permittivity. ``'lossless'`` and ``'perturbation'`` both solve the
        real eigenproblem here — they differ only in the post-processing
        :meth:`evaluate_qois` does. ``'lossy'`` solves the real problem *first*
        anyway, to seed the complex solve's shifts, then replaces the result with
        the complex one.

        The dielectric Q of the last solve is left on ``self._last_dielectric_q``
        (a list parallel to the returned frequencies, or None): the lossy path gets
        it from the eigenvalue, and it cannot be recomputed from the returned
        fields alone.
        """
        n_modes = self.requested_n_modes(n_modes=n_modes)
        system = self._build_system(mesh, mesh_p, m_pol, f_shift, direct_solver, materials)
        freq_fes, gfu_E, gfu_H = self._solve_system(system, n_modes, pinvit_maxit)
        self._last_dielectric_q = None
        if loss_model == 'lossy':
            freq_fes, gfu_E, gfu_H, self._last_dielectric_q = self._lossy_pass(
                mesh, mesh_p, m_pol, materials, freq_fes, n_modes,
                direct_solver=direct_solver, n_arnoldi=n_arnoldi)
        if save_dir:
            self.save_fields(save_dir, gfu_E, gfu_H, mesh_p, m_pol, freq_fes,
                             materials=materials)
        return freq_fes, gfu_E, gfu_H

    def _lossy_pass(self, mesh, mesh_p, m_pol, materials, freq_lossless, n_modes,
                    direct_solver=None, n_arnoldi=None):
        """Re-solve the (already-solved, lossless) problem with complex eps_r.

        Takes the lossless frequencies as the shift seeds and hands back
        ``(freq_fes, gfu_E, gfu_H, q_diel)``. Split out so the adaptive driver can
        refine on the cheap real problem and pay for the complex solve exactly
        once, on the finest mesh.
        """
        lam_shifts = [(2 * pi * f * 1e6 / c0) ** 2 for f in freq_lossless]
        return self._solve_lossy_system(mesh, mesh_p, m_pol, materials, lam_shifts,
                                        n_modes, direct_solver=direct_solver,
                                        n_arnoldi=n_arnoldi)

    @staticmethod
    def _mode_error_field(mesh, gfu, fes_rec, fes_rec_vec):
        """Per-element Zienkiewicz-Zhu recovery error field for a single mode.

        Both halves of the product-space solution are covered, so the estimator
        sees every mode:

        - the in-plane part through ``curl(u)`` (the azimuthal H) recovered into
          the continuous scalar space *fes_rec* — this is the whole error for a
          TM mode;
        - the azimuthal part through ``grad(u_phi)`` recovered into the vector
          space *fes_rec_vec* — this is the whole error for a **TE** mode, whose
          ``u`` (and hence ``curl(u)``) is identically zero.

        Each recovery gap is integrated element-wise as ``integral r*|f - proj|^2``
        and the two are summed. Returns a per-element numpy array (non-negative);
        the conjugates make it a magnitude on a complex (lossy) space too, where an
        un-conjugated square would be complex and its ``abs`` would not be an
        error norm.
        """
        u, uphi = gfu.components
        h = curl(u)
        hstar = GridFunction(fes_rec)
        hstar.Set(h)
        err = np.abs(Integrate(y * (h - hstar) * Conj(h - hstar), mesh, VOL,
                               element_wise=True).NumPy())

        g = grad(uphi)
        gstar = GridFunction(fes_rec_vec)
        gstar.Set(g)
        err = err + np.abs(Integrate(y * InnerProduct(g - gstar, g - gstar), mesh, VOL,
                                     element_wise=True).NumPy())
        return err

    def _error_fields(self, mesh, fes_rz, gfu_list):
        """Per-mode recovery error field for every mode in *gfu_list*.

        Returns a list of per-element numpy arrays (one per mode). Callers pass
        them to :meth:`_refinement_driver` for the marking field and take each
        one's max for the per-mode convergence tolerance.
        """
        _, fes_rec = fes_rz.CreateGradient()
        fes_rec_vec = VectorH1(mesh, order=fes_rz.globalorder,
                               complex=fes_rz.is_complex)
        return [self._mode_error_field(mesh, g, fes_rec, fes_rec_vec) for g in gfu_list]

    @staticmethod
    def _refinement_driver(fields):
        """Marking field for a mesh that has to resolve several modes at once.

        Each mode's error field is normalised by its **own** peak before the
        modes are combined (element-wise max). Combining the *raw* fields
        instead lets the high modes dominate — their errors are orders of
        magnitude larger — so a low mode's worst element never clears
        ``theta * max``, is never refined, and its error freezes at exactly the
        same value level after level (a flat step in an error-vs-DOF plot).

        Normalised, every mode's worst element scores 1.0 and is always marked,
        so no mode can stall. Returns a per-element array in [0, 1] (peak 1),
        which makes ``theta`` a fraction of *each mode's own* peak. None if
        there is no error to act on.
        """
        driver = None
        for f in fields:
            fmax = float(f.max()) if len(f) else 0.0
            if fmax <= 0:
                continue
            fn = f / fmax
            driver = fn if driver is None else np.maximum(driver, fn)
        return driver

    def _adaptive_refine_hcurl(self, mesh, mesh_p, n_modes, pinvit_maxit,
                               system, adaptive, first, save_dir=None):
        """Error-driven h-refinement of *mesh* (refined in place).

        Reuses the *system* (space + forms) built by :meth:`_build_system`
        so only one space is ever registered on the mesh. Starting from the
        *first* solve, mark-and-refine until every mode's recovery ``max_err``
        is below ``tol`` (or a DOF/refinement cap is hit), then return the
        solution on the finest mesh. Every level is recorded — DOF count,
        element count, the frequencies and the per-mode ``max_err`` — so a
        convergence study can plot error vs DOFs along the refinement path.
        """
        theta = float(adaptive['theta'])
        max_ndof = int(adaptive['max_ndof'])
        max_ref = int(adaptive['max_refinements'])
        tol = float(adaptive['tol'])
        fes, fes_rz = system['fes'], system['fes_rz']

        freq_fes, gfu_E, gfu_H = first
        history = []
        for step in range(max_ref + 1):
            fields = self._error_fields(mesh, fes_rz, gfu_E)
            per_mode_max = [float(f.max()) if len(f) else 0.0 for f in fields]
            # Drive and gate on the requested physical modes only. PINVIT solves
            # n_modes + 2 for accuracy; the top padding modes are barely converged
            # (huge, noisy error) and would otherwise hijack the refinement and
            # make the tolerance unreachable.
            n_use = min(n_modes, len(fields))
            gate_max = max(per_mode_max[:n_use]) if n_use else 0.0

            history.append({
                'refinement': step,
                'No of DOFs': int(fes.ndof),
                'No of Mesh Elements': int(mesh.GetNE(VorB.VOL)),
                'freq [MHz]': [float(f) for f in freq_fes],
                'max_err': per_mode_max,
            })

            # Stop once every requested mode is converged to tol, or a cap is hit.
            if fes.ndof >= max_ndof or step >= max_ref or gate_max < tol:
                break

            driver = self._refinement_driver(fields[:n_use])
            if driver is None:
                break
            # driver is normalised to peak 1 per mode, so theta is a fraction of
            # each mode's own peak — no mode can be crowded out of the marking.
            mesh.ngmesh.Elements2D().NumPy()["refine"] = (driver > theta)
            del gfu_E, gfu_H, fields, driver
            gc.collect()
            mesh.Refine()
            mesh.Curve(mesh_p)
            freq_fes, gfu_E, gfu_H = self._solve_system(system, n_modes, pinvit_maxit)

        self._last_adaptive_history = history
        if save_dir:
            with open(os.path.join(save_dir, 'adaptive_history.json'), 'w') as f:
                json.dump(history, f, indent=4, separators=(',', ': '))
        return freq_fes, gfu_E, gfu_H

    def _solve_eigenproblem(self, cav, save_dir, mesh, mesh_p, n_modes=None, m=0, adaptive=None,
                            materials=None, loss_model='lossless'):
        """Assemble and solve the Maxwell eigenvalue problem for azimuthal order
        *m*. Returns (freqs, E_fields, H_fields).

        When *adaptive* is a settings dict (see :meth:`_parse_adaptive`) the
        mesh is refined in place to resolve the requested modes — the recovery
        error estimator is polarisation-agnostic, so this drives refinement for
        any *m* — and the returned fields are those of the finest refinement.

        A ``'lossy'`` *loss_model* (complex permittivity, see
        :func:`resolve_loss_model`) runs the real problem first and then re-solves
        it complex, seeded by the lossless spectrum. Adaptive refinement therefore
        stays on the cheap real problem and the complex solve is paid for exactly
        once, on the finest mesh.
        """
        n_modes = self.requested_n_modes(cav, n_modes=n_modes)

        f_shift = 0
        direct_solver = default_direct_solver()
        pinvit_maxit = 20            # PINVIT iterations (P3-4: exposed via config)
        n_arnoldi = None
        if hasattr(cav, 'eigenmode_config') and cav.eigenmode_config:
            f_shift = cav.eigenmode_config.get('f_shift', 0)
            direct_solver = cav.eigenmode_config.get('direct_solver', direct_solver)
            pinvit_maxit = int(cav.eigenmode_config.get('pinvit_maxit', pinvit_maxit))
            n_arnoldi = cav.eigenmode_config.get('arnoldi_vectors', None)
        elif isinstance(cav, dict) and 'f_shift' in cav: # Fallback for some legacy calls
             f_shift = cav['f_shift']

        system = self._build_system(mesh, mesh_p, m, f_shift, direct_solver, materials)
        freq_fes, gfu_E, gfu_H = self._solve_system(system, n_modes, pinvit_maxit)

        self._last_adaptive_history = None
        if adaptive:
            freq_fes, gfu_E, gfu_H = self._adaptive_refine_hcurl(
                mesh, mesh_p, n_modes, pinvit_maxit, system,
                adaptive, first=(freq_fes, gfu_E, gfu_H), save_dir=save_dir)

        self._last_dielectric_q = None
        if loss_model == 'lossy':
            # Free the real solution first: the complex space alone is twice the
            # memory, and nothing below reads the lossless fields.
            del gfu_E, gfu_H, system
            gc.collect()
            # direct_solver is deliberately re-resolved rather than carried over
            # from the config: the shifted matrix here is complex symmetric and
            # indefinite, which not every backend that handles the real problem can
            # factorise.
            freq_fes, gfu_E, gfu_H, self._last_dielectric_q = self._lossy_pass(
                mesh, mesh_p, m, materials, freq_fes, n_modes,
                direct_solver=None, n_arnoldi=n_arnoldi)

        self.save_fields(save_dir, gfu_E, gfu_H, mesh_p, m, freq_fes, materials=materials)
        return freq_fes, gfu_E, gfu_H

    def solve_convergence(self, cav, eigenmode_config=None):
        """Adaptive-h convergence data: the full per-mode QOIs at every level.

        The mesh is refined adaptively (monopole-error driven, see
        :meth:`_parse_adaptive`). At **each** refinement level every requested
        polarisation is solved on the current mesh and the full QOIs of all its
        modes are evaluated — the same rich per-mode records
        :meth:`evaluate_qois` produces for a normal
        run. Each record is tagged with the refinement level (``h_pass``), the
        monopole DOF count at that level (``No of DOFs``) and the level solve
        time (``time [s]``). Returns a flat list of QOI dicts; nothing is
        written to disk.

        Only the *mesh* refinement is adaptive/monopole-driven; the m-pole
        solves reuse that mesh. Their fresh product spaces are released before
        each ``Refine()`` — leaving them registered makes netgen update stale
        spaces onto freed memory (an access violation).
        """
        eigenmode_config = eigenmode_config or {}
        mesh_config = eigenmode_config.get('mesh_config', {})
        mesh_h = mesh_h_metres(mesh_config)
        mesh_p = mesh_config.get('p', 3)
        adaptive = self._parse_adaptive(mesh_config) or self._parse_adaptive({'adaptive': True})
        theta = float(adaptive['theta'])
        max_ndof = int(adaptive['max_ndof'])
        max_ref = int(adaptive['max_refinements'])
        tol = float(adaptive['tol'])

        pols = parse_polarisations(eigenmode_config.get('polarisation', 0))
        n_modes = self.requested_n_modes(cav, eigenmode_config)
        conductivity = eigenmode_config.get('conductivity', SIGMA_COPPER)
        rs_ohm = eigenmode_config.get('surface_resistance', None)
        materials = self.resolve_materials(cav, eigenmode_config)
        L_mono = cav.parameters.get('L_m', None)
        if L_mono is None:
            L_mono = eigenmode_config.get('normalization_length', None)
        L_mpole = cav.parameters.get('L_m', 1)

        f_shift = eigenmode_config.get('f_shift', 0)
        direct_solver = eigenmode_config.get('direct_solver', default_direct_solver())
        pinvit_maxit = int(eigenmode_config.get('pinvit_maxit', 20))

        mesh = self._build_mesh(cav, mesh_h, mesh_p)
        system = self._build_system(mesh, mesh_p, 0, f_shift, direct_solver, materials)

        rows = []
        for level in range(max_ref + 1):
            t0 = time.perf_counter()
            freq_fes, gfu_E, gfu_H = self._solve_system(system, n_modes, pinvit_maxit)
            ndof_level = int(system['fes'].ndof)

            # Per-mode recovery error fields: each mode gets its OWN 'max_err' so
            # its convergence can be plotted individually. The refinement is
            # driven by (and the tolerance gates on) the requested physical
            # monopole modes only — the top PINVIT padding modes are barely
            # converged and would otherwise hijack the refinement.
            mono_err = self._error_fields(mesh, system['fes_rz'], gfu_E)
            n_use = min(n_modes, len(mono_err))
            driver = self._refinement_driver(mono_err[:n_use])
            gate_max = max((float(f.max()) for f in mono_err[:n_use]), default=0.0)

            level_rows = []
            if 0 in pols:
                for ii in range(len(freq_fes)):
                    q = self.evaluate_qois(mesh, gfu_E, gfu_H, freq_fes, mode_idx=ii,
                                           n_cells=cav.n_cells, L=L_mono,
                                           conductivity=conductivity, surface_resistance_ohm=rs_ohm,
                                           materials=materials)
                    # composite key '<m>-<mode>' — polarisation then mode index,
                    # e.g. '0-0' monopole mode 0, '1-0' dipole mode 0.
                    q['mode_index'] = f"{q['m']}-{ii}"
                    q['max_err'] = float(mono_err[ii].max()) if len(mono_err[ii]) else 0.0
                    level_rows.append(q)

            mpole_spaces = []
            for m_pol in [pp for pp in pols if pp > 0]:
                fr_m, gE_m, gH_m = self._solve_modes(mesh, mesh_p, m_pol, n_modes,
                                                     materials=materials)
                m_err = self._error_fields(mesh, gE_m[0].components[0].space, gE_m) if gE_m else []
                for ii in range(len(fr_m)):
                    q = self.evaluate_qois(mesh, gE_m, gH_m, fr_m, m_pol, mode_idx=ii,
                                           n_cells=cav.n_cells, L=L_mpole,
                                           conductivity=conductivity, surface_resistance_ohm=rs_ohm,
                                           materials=materials)
                    q['mode_index'] = f"{q['m']}-{ii}"
                    q['max_err'] = float(m_err[ii].max()) if len(m_err[ii]) else 0.0
                    level_rows.append(q)
                mpole_spaces.append((gE_m, gH_m))

            elapsed = time.perf_counter() - t0
            for q in level_rows:
                q['h_pass'] = level
                q['No of DOFs'] = ndof_level
                q['time [s]'] = elapsed
            rows.extend(level_rows)

            # stop once every requested monopole mode's max_err is below tol, or
            # a hard cap is hit (DOF count / refinement passes).
            if ndof_level >= max_ndof or level >= max_ref or gate_max < tol:
                break
            if driver is None:
                break
            # driver is normalised to peak 1 per mode, so theta is a fraction of
            # each mode's own peak — no mode can be crowded out of the marking.
            mesh.ngmesh.Elements2D().NumPy()["refine"] = (driver > theta)
            # Release this level's fresh (m-pole + monopole) grid functions so no
            # stale space is updated onto freed memory by Refine().
            del gfu_E, gfu_H, mpole_spaces, level_rows, mono_err, driver
            gc.collect()
            mesh.Refine()
            mesh.Curve(mesh_p)

        return rows

    # ──────────────────────────────────────────────────────────────────────
    # QOI evaluation
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _peak_field_per_material(mesh, u_gf, uphi_gf, materials):
        """``{'Epk_<material> [MV/m]': value}`` — peak |E| inside each material.

        Sampled at points pulled slightly off each element's vertices toward its
        centroid. A vertex *on* a material interface is ambiguous — E jumps there,
        so ``mesh(z, r)`` could return either side's value — while a point strictly
        inside the element belongs to exactly one material and still sits close
        enough to the boundary to capture the peak.
        """
        e_mag = sqrt(Norm(u_gf) ** 2
                     + Norm(uphi_gf * IfPos(y - AXIS_EPS, 1 / y, 0)) ** 2)
        out = {}
        for mat in sorted(set(materials)):
            best = 0.0
            for el in mesh.Materials(mat).Elements():
                vs = [mesh[v].point for v in el.vertices]
                cz = sum(p[0] for p in vs) / len(vs)
                cr = sum(p[1] for p in vs) / len(vs)
                for (pz, pr) in vs:
                    zz, rr = 0.95 * pz + 0.05 * cz, 0.95 * pr + 0.05 * cr
                    try:
                        best = max(best, e_mag(mesh(zz, rr)))
                    except Exception:
                        continue
            out[f'Epk_{mat} [MV/m]'] = best * 1e-6
        return out

    @staticmethod
    def evaluate_qois(mesh, gfu_E, gfu_H, freq_fes, m=0, beta=1, save_dir=None, mode_idx=1,
                      n_cells=1, L=1, conductivity=SIGMA_COPPER, surface_resistance_ohm=None,
                      write_axis=False, materials=None, loss_model=None, q_diel=None):
        """Cavity figures of merit for azimuthal order *m*, mode *mode_idx*.

        One evaluator for every polarisation — the eigenproblem is the unified
        product space (``u = (E_z, E_r)``, ``u_phi = r*E_phi``) for all m, and so
        are the QOIs here. Only the **voltage-derived** quantities branch on m:

        - m = 0 (monopole): the accelerating voltage is the on-axis integral of
          E_z and R/Q is longitudinal; the azimuthal integral gives 2*pi. TM
          modes (u_phi = 0) accelerate; TE modes (u = 0) get zero Vacc/R/Q.
        - m >= 1 (deflecting): E_z ~ r^m vanishes on axis, so the voltage is
          taken along an off-axis line r0 = aperture/2 and converted to the
          transverse kick by Panofsky-Wenzel, V_t = m V_z(r0)/(k r0); R/Q is
          transverse and 'R/Q_t' is normalised by r0^(2(m-1)) (r0-independent).
          The azimuthal integral gives pi.

        Everything else — stored energy, peak surface fields, the H1-projected
        surface-loss integral, Q, G, k_cc — is one expression with the
        polarisation's azimuthal factor. The wall material enters Q/Ploss/Rsh/G
        surface resistance (copper by default; pass
        *surface_resistance_ohm* for a fixed Rs, e.g. SRF niobium); G is
        material-independent.

        **Dielectric loss.** When a material carries a loss tangent, ``'Q []'`` is
        the *total* Q, ``1/Q = 1/Q_wall + 1/Q_diel``, and the two contributions are
        reported alongside it with ``'Q model'`` naming how Q_diel was obtained:

        - ``'perturbation'`` — from the volume integral
          ``P_diel = 0.5 w eps0 int eps'' |E|^2 dV`` over the *lossless* field,
          which is exactly the same bookkeeping as the wall loss;
        - ``'lossy'`` — from the complex eigenvalue of the full lossy solve,
          passed in as *q_diel*.

        The geometry factor G stays ``Q_wall * Rs``: it is a property of the wall
        and the mode shape, so folding dielectric loss into it would make a
        material-independent quantity depend on the filling.

        A run with no dielectric loss adds none of these keys, so its QOIs are
        exactly what they always were.
        """
        w = 2 * pi * freq_fes[mode_idx] * 1e6
        # NOTE: this c0 is the BEAM velocity in the transit-time factor, not the
        # wave speed in the medium. It stays c0 even with a dielectric present —
        # do not "correct" it to c0/sqrt(eps_r).
        k_wave = w / (beta * c0)
        eps_cf = material_cfs(mesh, materials)
        u_gf, uphi_gf = gfu_E[mode_idx].components
        H_inplane, H_phi = gfu_H[mode_idx]
        az = 2 * pi if m == 0 else pi          # azimuthal integral factor

        # Active length for the Eacc normalisation. Elliptical cavities pass L
        # (half-cell length, so the active length is 2*L*n_cells); guns/pillboxes
        # pass L=None, so fall back to the on-axis field extent.
        axis_nodes = np.array(list(get_boundary_nodes(mesh, 'AXI')))
        minz, maxz = axis_nodes[:, 0].min(), axis_nodes[:, 0].max()
        if L is None:
            L = (maxz - minz) * 1e3 / (2 * n_cells)  # -> 2*L*n_cells*1e-3 == active length [m]

        # Stored energy. |E|^2 = |E_inplane|^2 + |E_phi|^2 with E_phi = u_phi/r,
        # so r*|E|^2 = r*|u|^2 + |u_phi|^2/r. The phi integral gives 2*pi (m=0)
        # or pi (m>=1, cos^2/sin^2 average).
        #
        # With a dielectric, eps_r goes INSIDE the integral — it varies in space,
        # so it is not a prefactor. This is the load-bearing line of the whole
        # material extension: R/Q, Q, G, Rsh and GR/Q all inherit U, so weighting
        # it wrongly corrupts every derived figure of merit at once.
        # InnerProduct conjugates its second argument, so this is |u|^2 for a
        # complex (lossy) field and u.u for a real one — the same quantity either
        # way. u_phi is scalar, so its modulus is spelled out.
        e2_density = (y * InnerProduct(u_gf, u_gf)
                      + 1 / y * uphi_gf * Conj(uphi_gf))
        energy_density = e2_density if eps_cf is None else eps_cf * e2_density
        U = az * 0.5 * eps0 * Integrate(energy_density, mesh).real

        # Dielectric loss, perturbatively: P_diel = 0.5 w eps0 int eps'' |E|^2 dV is
        # the SAME volume integral as U with eps' -> eps'', so it inherits the
        # r-weighting and the azimuthal factor unchanged. eps_im_cf is None unless
        # some region has a loss tangent, which is what keeps a lossless run's QOIs
        # untouched.
        eps_im_cf = material_loss_cfs(mesh, materials)
        Pdiel_pert = (0.0 if eps_im_cf is None else
                      az * 0.5 * w * eps0 * Integrate(eps_im_cf * e2_density, mesh).real)

        norm_u = Norm(u_gf)
        xpnts_surf = np.array(list(get_boundary_nodes(mesh, 'PEC')))
        r0 = 0.5 * xpnts_surf[:, 1][xpnts_surf[:, 1] > AXIS_EPS].min()   # aperture/2 (m>=1)
        n_ax_pts = int(5000 * (maxz - minz))
        xpnts_ax = np.linspace(minz, maxz, n_ax_pts)

        # --- Voltage / gradient / R-over-Q: the only physics that branches -----
        # CONVENTION: R/Q here is the *linac* definition R/Q = V^2 / (w U), with U
        # the total stored energy. This is exactly TWICE the *circuit/accelerator*
        # definition R/Q = V^2 / (2 w U) that many references tabulate — e.g. the
        # TESLA design report lists R/Q = 518 Ohm (circuit), which is 1036 Ohm here
        # (linac). The reconstructed beam impedance divides it back (peak = 0.5 Q R/Q)
        # so the impedance is convention-consistent; only this QOI column is 2x a
        # circuit-convention table.
        if m == 0:
            # On-axis E_z; a TE mode has no E_z, so its Vacc is zero.
            Vout = abs(Integrate(u_gf[0] * exp(1j * k_wave * x), mesh,
                                 definedon=mesh.Boundaries('AXI')))
            RoQ_norm = RoQ = Vout ** 2 / (w * U)
            Ez_axis = np.array([norm_u(mesh(xi, 0.0)) for xi in xpnts_ax])   # |E| on axis
        else:
            # Off-axis line r0 + Panofsky-Wenzel kick (E_z ~ r^m vanishes on axis).
            Ez_r0 = np.array([u_gf(mesh(xi, r0))[0] for xi in xpnts_ax])
            trapz = getattr(np, 'trapezoid', np.trapz)
            Vz = abs(trapz(Ez_r0 * np.exp(1j * k_wave * xpnts_ax), xpnts_ax))
            Vout = m * Vz / (k_wave * r0)
            RoQ = Vout ** 2 / (w * U)
            RoQ_norm = RoQ / r0 ** (2 * (m - 1))
            Ez_axis = np.abs(Ez_r0)
        Eout = Vout / (n_cells * L * 1e-3 * 2)   # active length = 2*L*n_cells

        # --- Peak surface fields (pointwise maxima over the azimuth) -----------
        norm_H_in, norm_H_phi = Norm(H_inplane), Norm(H_phi)
        Esurf, Hsurf = [], []
        for (xi, yi) in xpnts_surf:
            mip = mesh(xi, yi)
            e_phi = abs(uphi_gf(mip)) / yi if yi > AXIS_EPS else 0.0
            Esurf.append(max(norm_u(mip), e_phi))
            Hsurf.append(max(norm_H_in(mip), norm_H_phi(mip)))
        Epk = max(Esurf)
        Hpk = max(Hsurf)

        # --- Surface power loss: H1-projected boundary integral (all m) --------
        # H = curl(E)/(mu0 w) cannot be SIMD-evaluated as a GridFunction curl on
        # a boundary, so it is projected into continuous H1 fields whose traces
        # integrate cleanly — mesh-convergent and higher-order than sampling |H|
        # at the element endpoints. 0.5 = time averaging, az = azimuth.
        Rs = surface_resistance(w, conductivity, surface_resistance_ohm)
        order_ = u_gf.space.globalorder
        Hphi_gf = GridFunction(H1(mesh, order=order_, complex=True))
        Hphi_gf.Set(H_phi)
        Hin_gf = GridFunction(VectorH1(mesh, order=order_, complex=True))
        Hin_gf.Set(H_inplane)
        Ploss = az * 0.5 * Rs * Integrate(
            y * (Hphi_gf * Conj(Hphi_gf) + InnerProduct(Hin_gf, Hin_gf)), mesh,
            definedon=mesh.Boundaries('PEC')).real

        # Cell-to-cell coupling: mode vs the lowest passband mode (index 0).
        if len(freq_fes) > 1:
            kcc = 2 * (freq_fes[mode_idx] - freq_fes[0]) / (freq_fes[mode_idx] + freq_fes[0]) * 100
        else:
            kcc = 0

        # Wall Q and the geometry factor: both are wall-only by definition, so
        # they are computed before any dielectric loss is folded in.
        Q_wall = w * U / Ploss
        G = Q_wall * Rs

        # Total Q. Q_diel comes from the complex eigenvalue on the lossy path and
        # from the perturbation integral otherwise; either way the wall loss is the
        # perturbative contribution added here (a Leontovich impedance BC would make
        # the eigenproblem nonlinear in w, which is not worth it for a wall).
        Q_diel = None
        if loss_model == 'lossy' and q_diel is not None and mode_idx < len(q_diel):
            # A non-finite Q_diel means the eigenvalue came back purely real, i.e.
            # this mode sees no loss at all. Report no dielectric loss rather than an
            # infinity, which JSON cannot round-trip.
            qd = float(q_diel[mode_idx])
            Q_diel = qd if np.isfinite(qd) and qd > 0 else None
            Pdiel = w * U / Q_diel if Q_diel else 0.0
        elif Pdiel_pert > 0:
            Pdiel = Pdiel_pert
            Q_diel = w * U / Pdiel
        else:
            Pdiel = 0.0
        Q = w * U / (Ploss + Pdiel)

        # Axis field flatness (min/max of the on-axis |E| peaks). ``distance``
        # must stay >= 1: an axis shorter than 20 mm gives n_ax_pts < 100, and
        # find_peaks then raises — killing the whole solve over a cosmetic QOI.
        peaks, _ = find_peaks(Ez_axis, distance=max(1, n_ax_pts // 100), width=100)
        try:
            ff = min(Ez_axis[peaks]) / max(Ez_axis[peaks]) * 100
        except ValueError:
            ff = 0

        qois = {
            "m": m,
            "polarisation": pol_name(m),
            "Normalization Length [mm]": 2 * L,
            "N Cells": n_cells,
            "freq [MHz]": freq_fes[mode_idx],
            "Q []": Q,
            "Vacc [MV]": Vout * 1e-6,
            "Eacc [MV/m]": Eout * 1e-6,
            "Epk [MV/m]": Epk * 1e-6,
            "Hpk [A/m]": Hpk,
            "Bpk [mT]": mu0 * Hpk * 1e3,
            "kcc [%]": kcc,
            "ff [%]": ff,
            "Rsh [MOhm]": RoQ * Q * 1e-6,
            "R/Q [Ohm]": RoQ,
            "Epk/Eacc []": Epk / Eout,
            "Bpk/Eacc [mT/MV/m]": mu0 * Hpk * 1e9 / Eout,
            "G [Ohm]": G,
            "GR/Q [Ohm^2]": G * RoQ,
            "U [J]": U,
            "Ploss [W]": Ploss,
            "No of Mesh Elements": mesh.GetNE(VorB.VOL),
        }
        if materials:
            # How this Q was obtained. A dielectric-loaded result without this tag
            # is ambiguous — the same cavity has three defensible Q values — so the
            # tag ships with every materials run, lossless ones included.
            qois["Q model"] = (loss_model if loss_model in ('lossless', 'perturbation', 'lossy')
                               else ('perturbation' if Q_diel is not None else 'lossless'))
            if Q_diel is not None:
                qois["Q_wall []"] = Q_wall
                qois["Q_diel []"] = Q_diel
                qois["Pdiel [W]"] = Pdiel
                qois["tan_delta []"] = max_tan_delta(materials)
            # Peak |E| inside each material. E is DISCONTINUOUS across a dielectric
            # interface (normal D is what is continuous), so the PEC-wall sample
            # above cannot see the in-dielectric peak — which is the field that
            # matters for breakdown in a dielectric-loaded structure.
            qois.update(NGSolveMEVP._peak_field_per_material(mesh, u_gf, uphi_gf,
                                                             materials))
        if m >= 1:
            # Transverse-specific keys (the shared Vacc/Eacc/R-Q above already
            # hold the transverse analogues for m>=1).
            qois["r0 [mm]"] = r0 * 1e3
            qois["Vt [MV]"] = Vout * 1e-6
            qois["Et [MV/m]"] = Eout * 1e-6
            qois["R/Q_t [Ohm/m^(2(m-1))]"] = RoQ_norm

        # Only the PRIMARY mode of interest owns the axis-field CSV the plots
        # read; without the write_axis gate the all-modes loop clobbered it
        # (the LAST mode won), so a single-cell fundamental's plot showed a
        # higher mode (a node at the centre instead of the accelerating peak).
        if save_dir and write_axis:
            if m == 0:
                pd.DataFrame({'z(0, 0)': xpnts_ax, '|Ez(0, 0)|': Ez_axis}).to_csv(
                    os.path.join(save_dir, 'Ez_0_abs.csv'),
                    index=False, sep='\t', float_format='%.32f')
            else:
                pd.DataFrame({'z': xpnts_ax, '|Ez(r0)|': Ez_axis}).to_csv(
                    os.path.join(save_dir, f'Ez_r0_abs_mode_{mode_idx}.csv'),
                    index=False, sep='\t', float_format='%.32f')

        return qois

    # ──────────────────────────────────────────────────────────────────────
    # Persistence
    # ──────────────────────────────────────────────────────────────────────

    def save_fields(self, project_folder, gfu_E, gfu_H, mesh_p, m, freqs,
                    geom_order=None, materials=None):
        """Persist the eigenmode fields so they reload consistently — including
        after **adaptive** refinement.

        An adaptively-refined NGSolve mesh does NOT round-trip: any reload (pickle
        or netgen ``.vol``) returns a *flattened* mesh whose space has a different
        ndof than the hierarchical one the field was solved on, so raw-pickling the
        GridFunctions gave the ``BaseVector::Set: size a != b`` error hit by
        show_fields / multipacting after an adaptive run. The fix: save the mesh as
        ``.vol``, **reload the flattened mesh here**, and project each mode's E onto
        the space built on it (the geometry and order are identical, so the
        projection is exact to machine precision). Save those projected coefficient
        vectors + the metadata needed to rebuild the space (``mesh_p``, azimuthal
        order ``m``, per-mode ``freqs``); ``gfu_H`` is derived from E on reload
        (see :meth:`load_fields`).

        *geom_order* is the mesh's **geometric curve order**, kept distinct from the
        field's FES order ``mesh_p``: the eigenmode solve curves to ``mesh_p`` (so it
        defaults there), but multipacting's own-field mesh is deliberately **straight**
        (``geom_order=1``) so the tracker's collision polyline coincides with the
        element edges — reload must not curve it (see ``solve_multipacting_field``)."""
        n = len(gfu_E)
        geom_order = int(mesh_p if geom_order is None else geom_order)
        # 'materials' records what the fields were solved in. At mu_r = 1 the H
        # envelopes reload correctly without it (H = curl(E)/(mu0 w) is exact), so
        # this is provenance plus the hook a future mu_r would need. Absent in old
        # caches -> None -> vacuum, so they stay loadable.
        # 'complex' records the arithmetic the fields were solved in: a lossy solve
        # returns complex coefficient vectors, and rebuilding them on a real space
        # would silently drop the phase (and the loss with it).
        is_complex = bool(n and gfu_E[0].space.is_complex)
        meta = {'mesh_p': int(mesh_p), 'm': int(m), 'n_modes': int(n),
                'geom_order': geom_order, 'freqs': [float(v) for v in freqs],
                'materials': materials, 'complex': is_complex}
        if n:
            # Persist the (round-tripping) flattened mesh, then reload it so the
            # saved vectors live on the exact space reload will rebuild.
            src_mesh = gfu_E[0].space.mesh
            self.save_mesh(project_folder, src_mesh)
            flat = self.load_mesh(project_folder)
            if geom_order > 1:
                flat.Curve(geom_order)
            fes = self._build_system(flat, mesh_p, m, materials=materials,
                                     complex_fes=is_complex)['fes']
            fes.Update()
            dtype = complex if is_complex else float
            vecs = []
            for g in gfu_E:
                fg = GridFunction(fes)
                fg.components[0].Set(g.components[0])       # HCurl E block
                fg.components[1].Set(g.components[1])       # H1 u_phi block
                vecs.append(np.asarray(fg.vec.FV().NumPy(), dtype=dtype).copy())
            np.save(os.path.join(project_folder, 'gfu_E_vecs.npy'), np.stack(vecs))
        else:
            np.save(os.path.join(project_folder, 'gfu_E_vecs.npy'), np.empty((0, 0)))
        with open(os.path.join(project_folder, 'field_meta.json'), 'w') as f:
            json.dump(meta, f, indent=2)

    @staticmethod
    def save_mesh(folder, mesh):
        """Save the mesh as netgen ``.vol`` into *folder*. Unlike a pickle, a
        ``.vol`` round-trips a refined mesh to a stable (flattened) mesh, which is
        what :meth:`save_fields` projects the fields onto and :meth:`load_fields`
        rebuilds — so the two always agree."""
        mesh.ngmesh.Save(os.path.join(folder, "mesh.vol"))

    @staticmethod
    def load_mesh(folder):
        """Load the mesh (netgen ``.vol``; legacy ``mesh.pkl`` fallback)."""
        vol = os.path.join(folder, 'mesh.vol')
        if os.path.exists(vol):
            return Mesh(vol)
        with open(os.path.join(folder, 'mesh.pkl'), 'rb') as f:
            return pickle.load(f)

    def load_fields(self, folder, mode):
        """Reload the eigenmode fields: rebuild the product space on the saved
        ``.vol`` mesh, load the projected E vectors, and reconstruct the H
        envelopes from E. Returns ``(gfu_E, gfu_H)`` in the shape
        :meth:`_solve_system` produces. Falls back to the legacy ``gfu_EH.pkl``
        for caches written before this format (fixed meshes only)."""
        meta_path = os.path.join(folder, 'field_meta.json')
        if not os.path.exists(meta_path):
            with open(os.path.join(folder, 'gfu_EH.pkl'), "rb") as f:
                [gfu_E, gfu_H] = pickle.load(f)
            return gfu_E, gfu_H

        with open(meta_path) as f:
            meta = json.load(f)
        mesh = self.load_mesh(folder)
        # Curve to the SAVED geometric order (mesh_p for the eigenmode mesh; 1 —
        # i.e. no curving — for multipacting's deliberately straight own-field mesh).
        geom_order = int(meta.get('geom_order', meta['mesh_p']))
        if geom_order > 1:
            mesh.Curve(geom_order)
        # Absent in caches written before the lossy path existed -> real, which is
        # what those caches are.
        fes = self._build_system(mesh, meta['mesh_p'], meta['m'],
                                 materials=meta.get('materials'),
                                 complex_fes=bool(meta.get('complex', False)))['fes']
        fes.Update()
        vecs = np.load(os.path.join(folder, 'gfu_E_vecs.npy'))
        m_pol = meta['m']
        inv_r = IfPos(y - AXIS_EPS, 1 / y, 0)
        gfu_E, gfu_H = [], []
        for i in range(meta['n_modes']):
            gfu = GridFunction(fes)
            fv = gfu.vec.FV().NumPy()
            if len(vecs[i]) != len(fv):
                raise ValueError(
                    f"eigenmode field reload: saved vector has {len(vecs[i])} DOFs "
                    f"but the rebuilt space has {len(fv)}.")
            fv[:] = vecs[i]
            gfu_E.append(gfu)
            u_gf, uphi_gf = gfu.components
            w = 2 * pi * meta['freqs'][i] * 1e6
            H_inplane = inv_r / (mu0 * w) * (m_pol * u_gf + grad(uphi_gf))
            H_phi = 1 / (mu0 * w) * curl(u_gf)
            gfu_H.append((H_inplane, H_phi))
        return gfu_E, gfu_H

    # ──────────────────────────────────────────────────────────────────────
    # Visualization
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _field_cf(gfu_e, gfu_h, which):
        """Magnitude coefficient function for plotting the product-space fields
        (every polarisation, monopole included).

        'E'/'H' give the total magnitude of the azimuthal envelope, combining
        the in-plane and azimuthal groups; 'Ephi'/'Hphi' isolate the azimuthal
        component. For a TM mode the azimuthal E vanishes and the in-plane H
        vanishes, so 'E' reduces to |E_inplane| and 'H' to |H_phi| — the same
        fields the HCurl-only monopole formulation used to plot.
        """
        u_gf, uphi_gf = gfu_e.components
        H_inplane, H_phi = gfu_h
        which = which.lower()
        e_phi = uphi_gf * IfPos(y - AXIS_EPS, 1 / y, 0)   # E_phi = u_phi / r
        if which in ('ephi', 'e_phi'):
            return Norm(e_phi)
        if which in ('hphi', 'h_phi'):
            return Norm(H_phi)
        if which == 'h':
            return sqrt(Norm(H_inplane) ** 2 + Norm(H_phi) ** 2)
        return sqrt(Norm(u_gf) ** 2 + Norm(e_phi) ** 2)

    def show_fields(self, folder, mode=1, which='E', plotter='ngsolve'):
        gfu_E, gfu_H = self.load_fields(folder, mode)
        # Draw over the field's OWN mesh (curved to the solve order inside
        # load_fields) so the field CF and the render mesh are the same object.
        mesh = gfu_E[mode].space.mesh
        field_cf = self._field_cf(gfu_E[mode], gfu_H[mode], which)

        if plotter == 'matplotlib':
            self._plot_field_matplotlib(mesh, field_cf)
        else:
            return Draw(field_cf, mesh, order=2, settings={'Objects': {'Wireframe': False}})

    def show_mesh(self, folder, plotter='ngsolve'):
        mesh = self.load_mesh(folder)

        if plotter == 'matplotlib':
            self._plot_mesh_matplotlib(mesh)
        else:
            return Draw(mesh)

    def show_geometry(self, cav, maxh=20e-3, order=1, plotter='ngsolve'):
        """Draw the cavity's meshed geometry *without* needing a prior run — a
        coarse mesh is built on the fly just to preview the analysed domain."""
        mesh = self._build_mesh(cav, maxh, order)
        if plotter == 'matplotlib':
            self._plot_mesh_matplotlib(mesh)
        else:
            return Draw(mesh)

    @staticmethod
    def _mesh_points_and_triangles(mesh, subdivide=0):
        """Build (points, triangle-index-array) from the mesh's *actual*
        element connectivity.

        Using the real triangles avoids the spurious elements a point-cloud
        Delaunay triangulation creates across concave regions (the beampipe
        gap, the axis), which the old code tried — and failed — to remove with
        a hardcoded, scale-dependent edge-length mask. ``subdivide`` levels of
        uniform refinement add edge/'face' points so the high-order field and
        the curved boundary render more smoothly.
        """
        verts = list(mesh.vertices)
        vidx = {v.nr: i for i, v in enumerate(verts)}
        pts = [tuple(v.point) for v in verts]

        triangles = []
        for el in mesh.Elements(VorB.VOL):
            vs = [vidx[v.nr] for v in el.vertices]
            if len(vs) == 3:
                triangles.append(vs)
            elif len(vs) == 4:  # quad -> two triangles
                triangles.append([vs[0], vs[1], vs[2]])
                triangles.append([vs[0], vs[2], vs[3]])

        for _ in range(subdivide):
            pts, triangles = NGSolveMEVP._subdivide_triangles(pts, triangles)

        pts = np.array(pts)
        return pts, np.array(triangles)

    @staticmethod
    def _subdivide_triangles(pts, triangles):
        """One level of 1->4 midpoint refinement of a triangle list."""
        pts = list(pts)
        mids = {}

        def midpoint(a, b):
            key = (a, b) if a < b else (b, a)
            if key not in mids:
                mids[key] = len(pts)
                pts.append(((pts[a][0] + pts[b][0]) / 2.0,
                            (pts[a][1] + pts[b][1]) / 2.0))
            return mids[key]

        new_tris = []
        for a, b, c in triangles:
            ab, bc, ca = midpoint(a, b), midpoint(b, c), midpoint(c, a)
            new_tris += [[a, ab, ca], [ab, b, bc], [ca, bc, c], [ab, bc, ca]]
        return pts, new_tris

    @staticmethod
    def _sample_field(mesh, pts, field_cf):
        """Evaluate *field_cf* at each (x, y) in *pts*, robust to boundary
        points that fall just outside curved elements and to 1/r envelopes
        that blow up on the axis."""
        vals = []
        for xv, yv in pts:
            try:
                val = field_cf(mesh(xv, yv))
            except Exception:
                try:
                    val = field_cf(mesh(xv, yv * (1 - 1e-9) if yv else 1e-9))
                except Exception:
                    val = np.nan
            vals.append(val if np.isfinite(val) else np.nan)
        return np.array(vals)

    @staticmethod
    def _plot_field_matplotlib(mesh, field_cf):
        """Render a field magnitude over the mesh using matplotlib tricontourf."""
        pts, triangles = NGSolveMEVP._mesh_points_and_triangles(mesh, subdivide=2)
        triang = tri.Triangulation(pts[:, 0], pts[:, 1], triangles=triangles)
        vals = NGSolveMEVP._sample_field(mesh, pts, field_cf)
        # Fill any residual NaNs (axis/boundary) so tricontourf spans the domain.
        if np.isnan(vals).any():
            vals = np.where(np.isnan(vals), np.nanmin(vals), vals)

        plt.tricontourf(triang, vals, levels=40, cmap='jet')
        plt.gca().set_aspect('equal', 'box')
        plt.show()

    @staticmethod
    def _plot_mesh_matplotlib(mesh):
        """Render the mesh using matplotlib triplot (true element edges)."""
        pts, triangles = NGSolveMEVP._mesh_points_and_triangles(mesh)
        triang = tri.Triangulation(pts[:, 0], pts[:, 1], triangles=triangles)
        plt.triplot(triang, lw=0.6, c='k')
        plt.gca().set_aspect('equal', 'box')
        plt.show()

    # ──────────────────────────────────────────────────────────────────────
    # Deformation utilities
    # ──────────────────────────────────────────────────────────────────────

    def gauss(self, n=11, sigma=1.0, shift=0.0):
        r = np.linspace(-int(n / 2) + 0.5, int(n / 2) - 0.5, n)
        g = 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-(r - shift) ** 2 / (2 * sigma ** 2))
        return g / max(g)

    def gaussian_deform(self, n_cells, surface, deformation_params):
        deform_vector = np.zeros((len(surface), 1))
        for n_cell in range(n_cells):
            n_cell_surf_indx = np.where(surface[:, 2] == n_cell + 1)[0]
            mn, mx = np.min(n_cell_surf_indx), np.max(n_cell_surf_indx)

            disp = deformation_params[n_cell * 3]
            sigma = (0.1 + (0.2 - 0.1) * (1 + deformation_params[n_cell * 3 + 1]) / 2) * len(n_cell_surf_indx)
            shift = deformation_params[n_cell * 3 + 2] * (mn + mx) / 2
            deform_vector += np.atleast_2d(disp * 2e-3 * self.gauss(len(surface), sigma, shift=shift)).T

        surface_def = deform_vector + surface
        surface_def[:, 2] = surface[:, 2]

        # enforce end planes
        surface_def[1, 0] = surface_def[0, 0]
        surface_def[-1, 0] = surface_def[-2, 0]
        surface_def[0, 1] = 0
        surface_def[-1, 1] = 0

        return pd.DataFrame(surface_def, columns=[1, 0, 2])


def get_boundary_nodes(mesh, boundary_name):
    """Extract unique boundary node coordinates for a named boundary."""
    boundary_nodes = set()
    for e in mesh.Elements(BND):
        if e.mat == boundary_name:
            for v in e.vertices:
                boundary_nodes.add(mesh[v].point)
    return boundary_nodes


class suppress_c_stdout_stderr:
    """Context manager to suppress C-level stdout/stderr (e.g. from Gmsh)."""

    def __enter__(self):
        self.stdout_fd = sys.__stdout__.fileno()
        self.stderr_fd = sys.__stderr__.fileno()
        self.saved_stdout_fd = os.dup(self.stdout_fd)
        self.saved_stderr_fd = os.dup(self.stderr_fd)
        self.null_fd = os.open(os.devnull, os.O_RDWR)
        os.dup2(self.null_fd, self.stdout_fd)
        os.dup2(self.null_fd, self.stderr_fd)

    def __exit__(self, exc_type, exc_val, exc_tb):
        os.dup2(self.saved_stdout_fd, self.stdout_fd)
        os.dup2(self.saved_stderr_fd, self.stderr_fd)
        os.close(self.null_fd)
        os.close(self.saved_stdout_fd)
        os.close(self.saved_stderr_fd)
