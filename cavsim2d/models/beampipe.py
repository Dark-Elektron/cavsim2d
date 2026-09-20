"""Straight circular beam pipe — the drift element of a beam line.

A :class:`Beampipe` is a cylinder of radius *R* and length *L*: a PEC barrel
closed at each end by an aperture. The apertures are built ``'PMC'``, the
natural (magnetic-wall) condition, so the element is **open** by default and the
eigenmode config steers it like any other geometry — ``boundary_conditions='ee'``
makes it a closed pillbox, ``'oo'`` puts a PML on each end.

That default is what makes the element composable: a beam pipe concatenated
between two devices must not carry a metal plate across the bore.
"""
import os
import warnings

from cavsim2d.models.base import Cavity
from cavsim2d.geometry import Profile


class Beampipe(Cavity):
    """A circular beam pipe of radius *R* and length *L*, in **mm**.

    Parameters
    ----------
    R : float
        Pipe radius, mm.
    L : float
        Pipe length, mm.
    ends : {'pmc', 'pec'} or (str, str), optional
        Wall type on the two end faces. ``'pmc'`` (default) leaves them as open
        apertures the solver can retag; ``'pec'`` builds them as solid metal
        plates, which no boundary condition can reopen. A 2-tuple sets the left
        and right ends independently.
    name : str, optional
        Element name, default ``'beampipe'``.

    Examples
    --------
    >>> bp = Beampipe(R=35.0, L=100.0)
    >>> bp.parameters
    {'R': 35.0, 'L': 100.0}
    """

    #: Passive: it carries no accelerating cell (see Cavity.contributes_cells).
    contributes_cells = False

    #: The end-face tags this element accepts.
    END_TAGS = {'pmc': 'PMC', 'pec': 'PEC'}

    def __init__(self, R, L, ends='pmc', name='beampipe', color='k',
                 plot_label=None, chain=1, spacing=None):
        super().__init__(n_cells=1, beampipe='none', name=name, color=color,
                         plot_label=plot_label, chain=chain, spacing=spacing)
        self.R = float(R)
        self.L = float(L)
        self.kind = 'beampipe'
        self.n_cells = 1
        self.n_modes = 1
        self.color = color
        self.ends = self._parse_ends(ends)
        self.beampipe = 'none'
        self.cell_parameterisation = 'simplecell'
        self.get_geometric_parameters()
        self.shape = {'IC': [self.R, self.L], 'BP': 'none'}
        self.shape_multicell = None

    @classmethod
    def _parse_ends(cls, ends):
        """Normalise *ends* to a ``('PMC'|'PEC', 'PMC'|'PEC')`` pair."""
        if isinstance(ends, str):
            ends = (ends, ends)
        try:
            left, right = ends
        except (TypeError, ValueError):
            raise ValueError(
                f"ends must be 'pmc', 'pec' or a (left, right) pair, got {ends!r}.")
        out = []
        for side, want in (('left', left), ('right', right)):
            key = str(want).lower()
            if key not in cls.END_TAGS:
                raise ValueError(
                    f"unknown {side} end {want!r}; use one of "
                    f"{sorted(cls.END_TAGS)}.")
            out.append(cls.END_TAGS[key])
        return tuple(out)

    def get_geometric_parameters(self):
        self.parameters = {'R': self.R, 'L': self.L}
        return self.parameters

    def create(self, n_cells=None, beampipe=None, mode=None):
        if self.projectDir:
            self.self_dir = os.path.join(self.projectDir, self.name)
            geo_dir = os.path.join(self.self_dir, 'geometry')
            os.makedirs(geo_dir, exist_ok=True)
            self.geo_filepath = os.path.join(geo_dir, 'geodata.geo')
            self.write_geometry(self.parameters, write=self.geo_filepath)
            self._write_geometry_snapshot()

    def profile(self):
        """Meridian boundary as a :class:`~cavsim2d.geometry.Profile` (metres).

        A rectangle in the (z, r) half-plane, centred on z = 0: the barrel is
        ``'PEC'``, each end face carries its :attr:`ends` tag, and the axis is
        ``'AXI'``.
        """
        try:
            R = float(self.parameters['R']) * 1e-3
            L = float(self.parameters['L']) * 1e-3
        except (KeyError, TypeError, ValueError):
            return None
        if R <= 0 or L <= 0:
            return None
        left, right = self.ends
        half = L / 2.0
        return self._chained(Profile(self.name)
                             .start(-half, 0.0)
                             .line_to(-half, R, left)      # left end face
                             .line_to(half, R, 'PEC')      # barrel
                             .line_to(half, 0.0, right)    # right end face
                             .close('AXI'))

    def write_geometry(self, parameters, n_cells=None, beampipe=None, write=None, **kwargs):
        """Write a gmsh ``.geo`` file for the pipe (the non-native fallback path)."""
        R_m = float(parameters['R']) * 1e-3
        L_m = float(parameters['L']) * 1e-3
        half = L_m / 2.0
        left, right = self.ends

        os.makedirs(os.path.dirname(write), exist_ok=True)
        with open(write.replace('.n', '.geo'), 'w') as cav:
            cav.write('SetFactory("OpenCASCADE");\n')
            cav.write(f"Point(1) = {{-{half:.16e}, 0, 0}};\n")
            cav.write(f"Point(2) = {{-{half:.16e}, {R_m:.16e}, 0}};\n")
            cav.write(f"Point(3) = {{{half:.16e}, {R_m:.16e}, 0}};\n")
            cav.write(f"Point(4) = {{{half:.16e}, 0, 0}};\n")
            cav.write("Line(1) = {1, 2};\n")     # left end face
            cav.write("Line(2) = {2, 3};\n")     # barrel
            cav.write("Line(3) = {3, 4};\n")     # right end face
            cav.write("Line(4) = {4, 1};\n")     # axis
            by_tag = {'PEC': ['2'], 'PMC': []}
            by_tag[left].append('1')
            by_tag[right].append('3')
            cav.write(f'\nPhysical Line("PEC") = {{{", ".join(sorted(by_tag["PEC"]))}}};\n')
            if by_tag['PMC']:
                cav.write(f'Physical Line("PMC") = {{{", ".join(sorted(by_tag["PMC"]))}}};\n')
            cav.write('Physical Line("AXI") = {4};\n')
            cav.write("\nCurve Loop(1) = {1, 2, 3, 4};\n")
            cav.write("Plane Surface(1) = {1};\n")
            cav.write("Reverse Surface 1;\n")
            cav.write('Physical Surface("Domain") = {1};\n')

    def rebuild(self, parameters, beampipe=None):
        """A fresh pipe from its ``(R, L)`` parameter dict."""
        return type(self)(float(parameters['R']), float(parameters['L']),
                          ends=tuple(t.lower() for t in self.ends),
                          name=self.name, color=self.color,
                          plot_label=self.plot_label,
                          chain=getattr(self, 'chain', 1),
                          spacing=getattr(self, 'spacing', None))

    def _reconstruct_state(self):
        state = super()._reconstruct_state()
        state['ends'] = list(self.ends)
        return state

    @classmethod
    def _reconstruct_from_state(cls, state):
        """Rebuild from disk. The generic stub carries only what the base class
        knows about, so the end tags — which :meth:`rebuild` reads — are put on
        it here, or reloading a saved pipe would fail on a missing ``ends``."""
        stub = cls.__new__(cls)
        stub.ends = tuple(state.get('ends', ('PMC', 'PMC')))
        stub.name = state.get('name', 'beampipe')
        stub.color = state.get('color', 'k')
        stub.plot_label = state.get('plot_label', None)
        stub.chain = state.get('chain', 1)
        stub.spacing = state.get('spacing', None)
        cav = stub.rebuild(dict(state['parameters']))
        cav.name = stub.name
        return cav


class CircularWaveguide(Beampipe):
    """Deprecated alias for :class:`Beampipe` with **PEC** end plates.

    Kept so existing scripts keep meshing the closed cylinder they were written
    against. New code should use ``Beampipe(R, L)`` — open by default, and
    steerable to a closed box with ``boundary_conditions='ee'``.
    """

    def __init__(self, R, L, ends='pec', **kwargs):
        warnings.warn(
            'CircularWaveguide is deprecated; use Beampipe(R, L). Note that the '
            "Beampipe ends are 'pmc' (open) by default, whereas the "
            "CircularWaveguide ends are 'pec' — pass boundary_conditions='ee' to "
            'the eigenmode config for the closed-box behaviour.',
            DeprecationWarning, stacklevel=2)
        kwargs.setdefault('name', 'circular_waveguide')
        super().__init__(R, L, ends=ends, **kwargs)
        self.kind = 'circular_waveguide'
