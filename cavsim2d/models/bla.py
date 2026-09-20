"""Beam line absorber (BLA) — a beam pipe with a lossy ring set into its wall.

A :class:`BLA` is a beam pipe of bore *R* whose wall steps **outward** over part
of its length to make room for a lossy dielectric ring: the standard way to damp
higher-order modes that propagate out of a cavity into the beam tube.

The bore is not obstructed. The ring's inner radius *is* the beam pipe radius,
and ``absorber_thickness`` is measured outward from it, so the absorber occupies
``R`` to ``R + absorber_thickness`` and the beam still sees a clear aperture of
radius *R*. That matches how the hardware is built — a SiC cylinder brazed into a
recess in the pipe wall, its inner surface flush with the bore.

The ring is a rectangular (annular) region, which is exactly the shape
:meth:`~cavsim2d.models.base.Cavity.add_dielectric` takes with ``z``/``r``, so it
rides the existing dielectric path: the eigenmode solver adds it to the profile
as a named material region at mesh time and applies the complex permittivity
from ``eigenmode_config['materials']``.
"""
import os

from cavsim2d.models.beampipe import Beampipe
from cavsim2d.geometry import Profile

#: Runs shorter than this (metres) are not emitted, so an absorber spanning the
#: full element does not produce a zero-length bore segment.
RUN_TOL = 1e-12


class BLA(Beampipe):
    """A beam pipe of bore *R* and length *L* with a lossy ring in its wall.

    All lengths are in **mm**.

    Parameters
    ----------
    R, L : float
        Bore radius and overall length. *R* is the free aperture the beam sees,
        and it is unchanged by the absorber.
    absorber_length : float
        Axial length of the lossy ring. Must not exceed *L*.
    absorber_thickness : float
        Radial thickness of the ring, measured **outward** from the bore, so the
        ring spans ``r = R`` to ``r = R + absorber_thickness`` and the wall steps
        out to accommodate it.
    eps_r : float, optional
        Real relative permittivity of the absorber, default 10.
    tan_delta : float, optional
        Loss tangent, default 0.3 — a lossy, ferrite-like load. Above
        ``loss_model``'s perturbative threshold the solver switches to the full
        complex eigenproblem on its own.
    z_absorber : float, optional
        Axial centre of the ring relative to the element centre, default 0.
    material : str, optional
        Region name, default ``'absorber'``. It is the key the solver looks the
        permittivity up by, and (prefixed with the element label) the material
        name in an :class:`~cavsim2d.models.assembly.Assembly`.
    maxh : float, optional
        Mesh size inside the ring, mm. A thin ring is crossed by a single
        element at the pipe-scale mesh size, so set this whenever
        ``absorber_thickness`` is small compared with the global ``maxh``.

    Examples
    --------
    >>> bla = BLA(R=35.0, L=150.0, absorber_length=80.0,
    ...           absorber_thickness=5.0, eps_r=10.0, tan_delta=0.3)
    >>> bla.dielectrics[0]['r']          # inner radius is the bore
    (35.0, 40.0)
    """

    #: Native-only. The inherited pipe writer would emit a plain cylinder, losing
    #: both the recess and the absorber region, and gmsh writes a single physical
    #: surface so it could not carry the sub-domain anyway. Setting this to None
    #: is how a model tells the generic clone/tune path it has no .geo writer.
    write_geometry = None

    def __init__(self, R, L, absorber_length, absorber_thickness,
                 eps_r=10.0, tan_delta=0.3, z_absorber=0.0,
                 material='absorber', maxh=None, color=(0.8, 0.3, 0.2),
                 ends='pmc', name='bla', plot_label=None,
                 chain=1, spacing=None):
        super().__init__(R, L, ends=ends, name=name, plot_label=plot_label,
                         chain=chain, spacing=spacing)
        self.kind = 'bla'
        self.material = str(material)
        self.absorber_maxh = None if maxh is None else float(maxh)
        self.absorber_color = tuple(float(c) for c in color)

        L_abs = float(absorber_length)
        t_abs = float(absorber_thickness)
        if not 0 < L_abs <= float(L) + 1e-9:
            raise ValueError(
                f'absorber_length must be in (0, L={L}] mm, got {absorber_length!r}.')
        if t_abs <= 0:
            raise ValueError(
                f'absorber_thickness must be positive, got {absorber_thickness!r}. '
                'It is measured outward from the bore, so it is not limited by R.')
        z_c = float(z_absorber)
        half = float(L) / 2.0
        if z_c - L_abs / 2 < -half - 1e-9 or z_c + L_abs / 2 > half + 1e-9:
            raise ValueError(
                f'the absorber ring (z = {z_c - L_abs / 2:.4g} to '
                f'{z_c + L_abs / 2:.4g} mm) does not fit inside the element '
                f'(z = {-half:.4g} to {half:.4g} mm).')

        self.parameters.update({
            'absorber_length': L_abs,
            'absorber_thickness': t_abs,
            'z_absorber': z_c,
        })
        self.add_dielectric(self.material, float(eps_r),
                            tan_delta=float(tan_delta),
                            z=(z_c - L_abs / 2, z_c + L_abs / 2),
                            r=(float(R), float(R) + t_abs),
                            maxh=self.absorber_maxh, color=self.absorber_color)

    # -- geometry ------------------------------------------------------------

    @property
    def absorber_span(self):
        """``(z_lo, z_hi)`` of the ring in element coordinates, mm."""
        p = self.parameters
        z_c, L_abs = float(p['z_absorber']), float(p['absorber_length'])
        return z_c - L_abs / 2.0, z_c + L_abs / 2.0

    def profile(self):
        """Meridian boundary as a :class:`~cavsim2d.geometry.Profile` (metres).

        A pipe of bore *R* whose wall steps out to ``R + absorber_thickness``
        over the ring, so the recess holding the absorber is outside the beam
        aperture. Both ends are apertures at the bore radius, so the element
        concatenates onto a plain pipe with no step at the junction.
        """
        try:
            p = self.parameters
            R = float(p['R']) * 1e-3
            L = float(p['L']) * 1e-3
            t = float(p['absorber_thickness']) * 1e-3
            z_lo, z_hi = (v * 1e-3 for v in self.absorber_span)
        except (KeyError, TypeError, ValueError):
            return None
        if R <= 0 or L <= 0 or t <= 0:
            return None

        left, right = self.ends
        half = L / 2.0
        R_out = R + t
        prof = Profile(self.name).start(-half, 0.0).line_to(-half, R, left)
        if z_lo + half > RUN_TOL:                 # bore run before the recess
            prof.line_to(z_lo, R, 'PEC')
        prof.line_to(z_lo, R_out, 'PEC')          # step out into the wall
        prof.line_to(z_hi, R_out, 'PEC')          # over the ring
        prof.line_to(z_hi, R, 'PEC')              # step back to the bore
        if half - z_hi > RUN_TOL:                 # bore run after the recess
            prof.line_to(half, R, 'PEC')
        prof.line_to(half, 0.0, right)
        return self._chained(prof.close('AXI'))

    def create(self, n_cells=None, beampipe=None, mode=None):
        """Provision the workspace. Native-only — no ``.geo`` is written."""
        if self.projectDir:
            self.self_dir = os.path.join(self.projectDir, self.name)
            os.makedirs(os.path.join(self.self_dir, 'geometry'), exist_ok=True)
            self._write_geometry_snapshot()

    # -- model plumbing ------------------------------------------------------

    def rebuild(self, parameters, beampipe=None):
        """A fresh BLA from its parameter dict, carrying the absorber material."""
        d = self.dielectrics[0] if self.dielectrics else {}
        return type(self)(
            R=float(parameters['R']), L=float(parameters['L']),
            absorber_length=float(parameters['absorber_length']),
            absorber_thickness=float(parameters['absorber_thickness']),
            eps_r=d.get('eps_r', 10.0), tan_delta=d.get('tan_delta', 0.3),
            z_absorber=float(parameters.get('z_absorber', 0.0)),
            material=self.material, maxh=self.absorber_maxh,
            color=self.absorber_color,
            ends=tuple(t.lower() for t in self.ends),
            name=self.name, plot_label=self.plot_label,
            chain=getattr(self, 'chain', 1), spacing=getattr(self, 'spacing', None))

    @classmethod
    def _reconstruct_from_state(cls, state):
        """Rebuild from disk. The absorber is created by ``__init__``, so the
        saved dielectric entry supplies its material properties rather than
        being re-added on top (which would collide with the one just made)."""
        d = (state.get('dielectrics') or [{}])[0]
        p = dict(state['parameters'])
        return cls(R=float(p['R']), L=float(p['L']),
                   absorber_length=float(p['absorber_length']),
                   absorber_thickness=float(p['absorber_thickness']),
                   eps_r=d.get('eps_r', 10.0), tan_delta=d.get('tan_delta', 0.3),
                   z_absorber=float(p.get('z_absorber', 0.0)),
                   material=state.get('material', 'absorber'),
                   maxh=d.get('maxh'),
                   color=tuple(d.get('color', (0.8, 0.3, 0.2))),
                   ends=tuple(t.lower() for t in state.get('ends', ('pmc', 'pmc'))),
                   name=state.get('name', 'bla'))

    def _reconstruct_state(self):
        state = super()._reconstruct_state()
        state['material'] = self.material
        return state
