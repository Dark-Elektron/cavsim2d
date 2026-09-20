"""Assembly — several devices concatenated into one simulatable beam line.

An :class:`Assembly` joins any number of elements (a beam pipe, a bellows, a
cavity, an absorber, or another assembly) into a single connected structure and
*is itself* a :class:`~cavsim2d.models.base.Cavity`: it meshes, runs eigenmode
and wakefield analyses, tunes, and takes part in optimisation and UQ like any
built-in geometry::

    line = (Beampipe(R=35, L=100)
            + Bellows(Ri=35, A=8, L_p=6, N_conv=10, R_root=1.2, R_crest=1.2)
            + EllipticalCavity(2, MIDCELL, beampipe='none')
            + Bellows(Ri=35, A=8, L_p=6, N_conv=10, R_root=1.2, R_crest=1.2)
            + BLA(R=35, L=150, absorber_length=80, absorber_thickness=5))

    line.eigenmode.run()

Parameters are namespaced by element label, ``'<label>:<name>'``, which is the
convention dielectric variables already use. So ``line.parameters['bellows_1:A']``
is the first bellows' convolution depth, and that same string is what a tune,
optimisation or UQ config names. The labels come from the elements' own ``name``,
suffixed ``_1``, ``_2``, ... where a name repeats.

Boundaries
----------
Concatenation *removes* the internal apertures: adjacent walls join directly, so
there is no interface boundary to carry a condition — the vacuum is one region.
Only the two outer ends remain, and :attr:`ends` overrides whatever the end
elements declared, in both directions (it can reopen a ``'pec'`` end, which the
solver's own ``boundary_conditions`` cannot).
"""
import importlib
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np

from cavsim2d.models.base import Cavity
from cavsim2d.models.beampipe import Beampipe


#: Fixed hue order for :meth:`Assembly.shade_elements`, drawn from the house
#: palette. The order is a farthest-point traversal of WARM under OKLab Delta E
#: (normal vision and Machado protan/deutan), measured on the *composited* band
#: colour rather than the source hue — a band is a wash over the surface, and
#: fading a saturated colour toward white collapses the hues together. Taking
#: WARM in its natural order left the worst pair at Delta E 1.2 (0.6 under CVD),
#: i.e. indistinguishable; this order holds every pair of a typical beam line at
#: >= 4.5 (3.5 under CVD). Slots are assigned in order and never cycled: past the
#: end, kinds fall back to a neutral and are told apart by their in-band label.
BAND_COLORS = ('#457B9D', '#E9C46A', '#8B4000', '#8A9A5B',
               '#264653', '#5B7553', '#3D5A45', '#6FA88F')

#: Neutral for element kinds beyond BAND_COLORS (see the note there).
BAND_OVERFLOW = '#9A9A94'


class Assembly(Cavity):
    """A beam line built by concatenating *elements*.

    Parameters
    ----------
    elements : sequence of Cavity
        The devices, ordered upstream to downstream. A nested ``Assembly`` is
        flattened into its own elements.
    gaps : float or sequence of float, optional
        Straight drift inserted at each junction, **mm**. A scalar sets every
        junction; a sequence sets each of the ``len(elements) - 1`` junctions and
        must have that length. Default 0 — the elements butt, and any drift is
        an explicit :class:`~cavsim2d.models.beampipe.Beampipe` element.
    ends : {'pmc', 'pec'} or (str, str), optional
        Override the wall type of the assembly's two outer end faces. ``None``
        (default) keeps whatever the first and last elements declare.
    allow_steps : bool, optional
        Set True when the line is meant to contain abrupt changes of bore, to
        silence the mismatch warning. Default False: a junction whose apertures
        differ is still built as a radial step, but :meth:`check_continuity`
        reports where each one is, so an accidental one is visible.
    name : str, optional
        Assembly name, default ``'assembly'``.

    Examples
    --------
    >>> line = Assembly([Beampipe(R=35, L=50), Beampipe(R=35, L=80)])
    >>> sorted(line.parameters)
    ['beampipe_1:L', 'beampipe_1:R', 'beampipe_2:L', 'beampipe_2:R']
    """

    def __init__(self, elements, gaps=None, ends=None, allow_steps=False,
                 name='assembly', color='k', plot_label=None):
        super().__init__(n_cells=1, beampipe='none', name=name, color=color,
                         plot_label=plot_label)
        self.kind = 'assembly'
        self.color = color

        flat = []
        for el in elements:
            if isinstance(el, Assembly):
                flat.extend(el.elements)
            else:
                flat.append(el)
        if len(flat) < 2:
            raise ValueError(
                'an Assembly needs at least two elements; a single device is '
                'already simulatable on its own.')
        for el in flat:
            if not callable(getattr(el, 'profile', None)):
                raise TypeError(
                    f'{type(el).__name__} {getattr(el, "name", "?")!r} has no '
                    'profile(), so it cannot be concatenated. Only models that '
                    'build a native Profile can take part in an Assembly.')
        self.elements = flat
        self.labels = self._make_labels(flat)
        self.gaps = self._normalise_gaps(gaps, len(flat))
        self.ends = None if ends is None else Beampipe._parse_ends(ends)
        self.allow_steps = bool(allow_steps)

        # Only the elements that actually carry accelerating cells count. This
        # sum sets the monopole mode of interest (the pi-mode index), so letting
        # a beam pipe or a bellows contribute would silently report the wrong
        # mode of the assembled line.
        self.n_cells = max(1, int(sum(int(getattr(el, 'n_cells', 1) or 1)
                                      for el in flat
                                      if getattr(el, 'contributes_cells', True))))
        self.n_modes = max(int(getattr(el, 'n_modes', 1) or 1) for el in flat)
        self.beampipe = 'none'
        self.cell_parameterisation = 'simplecell'
        self.check_continuity()
        self.pull()
        self.shape = {'IC': list(self.labels), 'BP': 'none'}
        self.shape_multicell = None

    # -- element bookkeeping -------------------------------------------------

    @staticmethod
    def _make_labels(elements):
        """One unique, stable label per element, from the elements' own names.

        A name used once is its own label; a name used more than once gets a
        ``_1``, ``_2``, ... suffix in order, so two bellows become ``bellows_1``
        and ``bellows_2`` while a lone beam pipe stays ``beampipe``.
        """
        names = [str(getattr(el, 'name', None) or type(el).__name__.lower())
                 for el in elements]
        for nm in names:
            if ':' in nm:
                raise ValueError(
                    f"element name {nm!r} contains ':', which separates the "
                    'element label from the parameter name; rename it with '
                    'set_name().')
        counts = {nm: names.count(nm) for nm in set(names)}
        seen = {}
        labels = []
        for nm in names:
            if counts[nm] == 1:
                labels.append(nm)
            else:
                seen[nm] = seen.get(nm, 0) + 1
                labels.append(f'{nm}_{seen[nm]}')
        return labels

    @staticmethod
    def _normalise_gaps(gaps, n_elements):
        """The ``n_elements - 1`` junction drifts, in mm."""
        n_gaps = n_elements - 1
        if gaps is None:
            return [0.0] * n_gaps
        try:
            out = [float(g) for g in gaps]
        except TypeError:
            out = [float(gaps)] * n_gaps
        if len(out) == 1:
            out = out * n_gaps
        if len(out) != n_gaps:
            raise ValueError(
                f'gaps must have one entry per junction: {n_gaps} for '
                f'{n_elements} elements, got {len(out)}.')
        if any(g < 0 for g in out):
            raise ValueError('gaps must be non-negative.')
        return out

    def element(self, label):
        """The element carrying *label*."""
        try:
            return self.elements[self.labels.index(label)]
        except ValueError:
            raise KeyError(
                f'no element labelled {label!r} in assembly {self.name!r}; '
                f'it has {self.labels}.')

    def check_continuity(self):
        """Report junctions where adjacent elements meet at different apertures.

        A change of bore between two elements is closed with an abrupt radial
        step. That is a legitimate structure — a collimator, a flange, a real
        cross-section change — so it is built, not refused. It is also easy to
        create by accident and invisible once the walls are joined, so every
        mismatch is reported in a single :class:`UserWarning` naming the two
        elements, their radii, the step size and where along the line it sits.

        Pass ``allow_steps=True`` when the steps are intended, to silence it.

        Elements whose ``profile()`` is ``None`` (degenerate parameters) are
        skipped; those are reported by whatever needs the geometry.

        Returns
        -------
        list
            One ``dict`` per mismatch, with ``index``, ``labels``, ``radii``
            (mm), ``step`` (mm) and ``z`` (mm, in assembly coordinates). Empty
            when every junction matches.
        """
        profiles = [el.profile() for el in self.elements]
        if any(p is None for p in profiles):
            return []
        try:
            offsets = self._offsets_from(profiles)
        except ValueError:
            return []

        found = []
        for i in range(len(profiles) - 1):
            left, right = profiles[i], profiles[i + 1]
            try:
                _, _, l_ap, _ = left._meridian_parts()
                r_ap, _, _, _ = right._meridian_parts()
            except ValueError:
                continue
            r_out = left._pts[l_ap['i0']][1] * 1e3
            r_in = right._pts[r_ap['i1']][1] * 1e3
            if abs(r_out - r_in) <= 1e-6:
                continue
            found.append({
                'index': i,
                'labels': (self.labels[i], self.labels[i + 1]),
                'radii': (r_out, r_in),
                'step': r_in - r_out,
                'z': offsets[i] + left.wall_span()[1] * 1e3 + self.gaps[i],
            })

        if found and not self.allow_steps:
            lines = [
                f"  z = {f['z']:9.3f} mm : {f['labels'][0]!r} ends at "
                f"r = {f['radii'][0]:.6g} mm, {f['labels'][1]!r} starts at "
                f"r = {f['radii'][1]:.6g} mm  ->  step of {f['step']:+.6g} mm"
                for f in found
            ]
            many = len(found) > 1
            header = (f'assembly {self.name!r}: aperture mismatch at '
                      f"{len(found)} junction{'s' if many else ''}, closed "
                      f"with {'abrupt radial steps' if many else 'an abrupt radial step'}:")
            # ASCII only: this lands on a terminal, and a cp1252 console
            # mangles an em dash into a replacement character.
            footer = ('If that is intended, pass allow_steps=True to silence '
                      'this. If not, insert a Taper between the elements, '
                      f"e.g. Taper(R_left={found[0]['radii'][0]:.6g}, "
                      f"R_right={found[0]['radii'][1]:.6g}, L=...).")
            warnings.warn('\n'.join([header] + lines + [footer]),
                          UserWarning, stacklevel=3)
        return found

    def set_ends(self, left, right=None):
        """Override the wall type on the assembly's two outer end faces.

        ``'pmc'`` leaves an end open (the solver can then retag it via
        ``boundary_conditions``); ``'pec'`` makes it a solid plate. This wins
        over whatever the first and last elements declare, so a
        ``Beampipe(ends='pec')`` used at the end of a line can still be opened.
        Internal junctions are unaffected — they have no boundary at all.
        """
        self.ends = Beampipe._parse_ends(left if right is None else (left, right))
        return self

    # -- parameters ----------------------------------------------------------

    def _key(self, label, name):
        return f'{label}:{name}'

    def _split(self, key):
        """``('<label>', '<rest>')`` for a namespaced parameter key."""
        label, sep, rest = str(key).partition(':')
        if not sep or label not in self.labels:
            raise ValueError(
                f'{key!r} is not an assembly variable. Assembly variables are '
                f"'<label>:<name>' with the label one of {self.labels}.")
        return label, rest

    def pull(self):
        """Refresh ``self.parameters`` from the elements' own parameter dicts."""
        merged = {}
        for label, el in zip(self.labels, self.elements):
            for key, value in (el.parameters or {}).items():
                merged[self._key(label, key)] = value
        self.parameters = merged
        return self.parameters

    def push(self):
        """Write ``self.parameters`` back into the elements.

        The tuner and the optimiser mutate ``self.parameters`` in place, so the
        elements are only correct once this has run; :meth:`profile` calls it
        first. Elements that validate their own parameters (a bellows' corner
        radii, say) are re-checked here, so an infeasible search point fails with
        the element's own named constraint.
        """
        for label, el in zip(self.labels, self.elements):
            prefix = f'{label}:'
            for key, value in self.parameters.items():
                if key.startswith(prefix):
                    name = key[len(prefix):]
                    if name in (el.parameters or {}):
                        el.parameters[name] = value
            check = getattr(el, 'check_feasible', None)
            if callable(check):
                check()
        return self

    def tune_variables(self):
        """Every element's tune variables, namespaced ``'<label>:<name>'``."""
        names = set()
        for label, el in zip(self.labels, self.elements):
            for v in el.tune_variables():
                names.add(self._key(label, v))
        return names

    def expand_variable(self, name):
        """Every parameter slot *name* refers to, namespaced.

        Delegated to the owning element, so ``'cav:Req'`` on an elliptical
        cavity still expands to its three per-cell slots — ``'cav:Req_m'``,
        ``'cav:Req_el'``, ``'cav:Req_er'``.
        """
        label, rest = self._split(name)
        el = self.element(label)
        return [self._key(label, slot) for slot in el.expand_variable(rest)]

    @staticmethod
    def _slots(el, name):
        """The element's own parameter slot(s) that *name* refers to.

        A bare name on a model that uses per-cell suffixes (``'Req'`` on an
        elliptical cavity) is not a slot of its own, so it is expanded through
        the element's own :meth:`~cavsim2d.models.base.Cavity.expand_variable` —
        the documented meaning being "the same quantity in every cell". The
        tuner performs that mapping itself for a bare cavity, but it keys off
        the *cavity's* ``uses_cell_suffixes``, and an assembly can hold several
        models with different conventions; asking each element is what keeps a
        mixed line correct.
        """
        if name in (el.parameters or {}) or el._dielectric_slot(name) is not None:
            return [name]
        return el.expand_variable(name)

    def get_tune_value(self, name):
        label, rest = self._split(name)
        el = self.element(label)
        slots = self._slots(el, rest)
        if len(slots) == 1:
            return el.get_tune_value(slots[0])
        return float(np.mean([float(el.get_tune_value(s)) for s in slots]))

    def set_tune_value(self, name, value):
        label, rest = self._split(name)
        el = self.element(label)
        for slot in self._slots(el, rest):
            el.set_tune_value(slot, value)
        self.pull()

    def _row_slots(self, key):
        """Resolve a spawn / optimisation column to this line's parameter slots.

        Bounds are written the way the user names them (``'cav:A'``), but an
        elliptical cavity stores ``A_m`` / ``A_el`` / ``A_er``. Without this the
        base filter drops the column and every candidate in the search evaluates
        the *same* geometry — the objectives then differ only by solver noise,
        which looks like a converged optimisation.
        """
        if key in (self.parameters or {}):
            return [key]
        try:
            label, rest = self._split(key)
            el = self.element(label)
            return [self._key(label, slot) for slot in self._slots(el, rest)]
        except (ValueError, KeyError):
            return []

    def get_geometric_parameters(self):
        return self.pull()

    # -- dielectrics ---------------------------------------------------------

    @property
    def dielectrics(self):
        """The elements' dielectric regions, translated into assembly coordinates.

        A region's ``z`` extent is absolute and in mm, so it has to travel with
        the element the concatenation moved. Material names are prefixed with
        the element label, which keeps two absorbers in one line distinct (they
        become separate mesh materials, and separate
        ``eigenmode_config['materials']`` keys).
        """
        out = []
        for label, el, dz in zip(self.labels, self.elements, self.offsets()):
            for d in (getattr(el, 'dielectrics', None) or ()):
                entry = dict(d)
                entry['material'] = f'{label}_{d["material"]}'
                if d.get('points'):
                    entry['points'] = [(z + dz, r) for z, r in d['points']]
                elif d.get('z') is not None:
                    entry['z'] = (d['z'][0] + dz, d['z'][1] + dz)
                out.append(entry)
        return out

    def add_dielectric(self, *args, **kwargs):
        raise NotImplementedError(
            f'add a dielectric to the element that carries it, not to the '
            f'assembly: e.g. assembly.element({self.labels[0]!r}).add_dielectric(...). '
            'The assembly collects its elements\' regions and translates them '
            'into assembly coordinates automatically.')

    def clear_dielectrics(self):
        for el in self.elements:
            el.clear_dielectrics()
        return self

    def _carry_dielectrics_to(self, clone):
        """Nothing to carry: an assembly's regions are *derived* from its
        elements, and a clone is built by rebuilding those elements, so it comes
        out already loaded. The base implementation appends onto
        ``clone.dielectrics``, which here is a freshly computed list, so it would
        be a silent no-op — this override says so rather than leaving it to be
        rediscovered."""
        return clone

    def dielectric_variables(self):
        names = set()
        for label, el in zip(self.labels, self.elements):
            for v in el.dielectric_variables():
                names.add(self._key(label, v))
        return names

    # -- geometry ------------------------------------------------------------

    def offsets(self):
        """Axial shift applied to each element's own profile, in **mm**.

        ``offsets()[i]`` is what element *i* was translated by to land in the
        assembled profile, so a consumer can map an element's own coordinates
        into assembly coordinates.
        """
        profiles = [el.profile() for el in self.elements]
        if any(p is None for p in profiles):
            return [0.0] * len(self.elements)
        return self._offsets_from(profiles)

    def _offsets_from(self, profiles):
        """Offsets in mm, given each element's already-built profile."""
        out = [0.0]
        z_end = profiles[0].wall_span()[1]
        for prof, gap in zip(profiles[1:], self.gaps):
            z_start = prof.wall_span()[0]
            dz = (z_end + gap * 1e-3) - z_start
            out.append(dz * 1e3)
            z_end = prof.wall_span()[1] + dz
        return out

    def element_spans(self):
        """Where each element sits along the line: ``[{...}, ...]`` in **mm**.

        One entry per element, with ``label``, ``kind`` (the model's own kind
        string, so two cavities share one), ``z0`` and ``z1``. Elements whose
        ``profile()`` is degenerate are skipped.
        """
        out = []
        for label, el, dz in zip(self.labels, self.elements, self.offsets()):
            prof = el.profile()
            if prof is None:
                continue
            zs = [z for z, _ in prof.points]
            out.append({
                'label': label,
                'kind': str(getattr(el, 'kind', None) or type(el).__name__),
                'z0': dz + min(zs) * 1e3,
                'z1': dz + max(zs) * 1e3,
            })
        return out

    def shade_elements(self, ax=None, alpha=0.28, legend=False, colors=None,
                       center=False, zorder=0, labels=True, edges=True):
        """Shade the axial extent of each element behind an existing plot.

        Bands of colour along z showing which device occupies which stretch of
        the line, so a mode plot or a field profile can be read against the
        hardware. Elements of the **same kind share a colour** — both cavities
        one shade, both tapers another — so a long line stays legible.

        Parameters
        ----------
        ax : matplotlib Axes, optional
            Axes to shade; a new one is made when omitted.
        alpha : float, optional
            Band opacity, default 0.28. Bands are washes, and a wash loses hue
            separation fast as it lightens: at 0.13 the best available hues are
            only Delta E ~1 apart, which is no separation at all. 0.28 keeps them
            apart while staying recessive under a curve.
        legend : bool, optional
            Add one legend entry per element *kind* (not per element). Off by
            default because *labels* is on: a light band cannot carry identity
            by colour alone, so every band is named in place instead.
        colors : dict, optional
            ``{kind: color}`` overrides; kinds not listed fall back to the house
            palette, assigned in order of first appearance.
        center : bool, optional
            Set True when shading a plot drawn by ``plot('geometry')``, which
            centres the line on z = 0 by default. The axis-field plots use the
            uncentred frame, so the default False matches them.
        labels : bool, optional
            Write each element's label inside its band, default True. Colour
            groups the bands by kind; the label is what identifies them, which
            is what keeps the figure readable in greyscale and for a CVD reader.
        edges : bool, optional
            Draw a faint rule at every element boundary, default True. Two
            adjacent elements of the same kind share a colour and would
            otherwise merge into one band — two cavities in a row being the
            obvious case.

        Returns
        -------
        matplotlib.axes.Axes

        Examples
        --------
        >>> ax = line.plot_axis_field(mode=3, show=False)      # doctest: +SKIP
        >>> line.shade_elements(ax)                            # doctest: +SKIP
        """
        spans = self.element_spans()
        if not spans:
            return ax
        if ax is None:
            _, ax = plt.subplots(figsize=(11, 3))

        shift = 0.0
        if center:
            lo = min(s['z0'] for s in spans)
            hi = max(s['z1'] for s in spans)
            shift = 0.5 * (lo + hi)

        palette = {}
        for span in spans:                       # first appearance takes the slot
            kind = span['kind']
            if kind not in palette:
                n = len(palette)
                palette[kind] = (BAND_COLORS[n] if n < len(BAND_COLORS)
                                 else BAND_OVERFLOW)
        if colors:
            palette.update(colors)

        seen = set()
        for span in spans:
            kind = span['kind']
            label = None
            if legend and not labels and kind not in seen:
                label = kind
                seen.add(kind)
            ax.axvspan(span['z0'] - shift, span['z1'] - shift,
                       color=palette[kind], alpha=alpha, lw=0, zorder=zorder,
                       label=label)
            if labels:
                ax.annotate(span['label'],
                            xy=(0.5 * (span['z0'] + span['z1']) - shift, 0.98),
                            xycoords=('data', 'axes fraction'),
                            ha='center', va='top', fontsize=7, rotation=90,
                            color='0.35')

        if edges:
            for z in sorted({s['z0'] for s in spans} | {s['z1'] for s in spans}):
                ax.axvline(z - shift, color='0.55', lw=0.6, ls=(0, (4, 3)),
                           zorder=zorder + 1)
        return ax

    def profile(self):
        """The assembled meridian as one :class:`~cavsim2d.geometry.Profile`.

        The elements' walls are joined in order, the internal apertures dropped,
        and the two outer ends retagged when :attr:`ends` is set. Returns
        ``None`` if any element's own profile is degenerate, matching the
        convention the single-device models use.
        """
        self.push()
        profiles = [el.profile() for el in self.elements]
        if any(p is None for p in profiles):
            return None

        prof = profiles[0]
        for nxt, gap in zip(profiles[1:], self.gaps):
            # Mismatches were already reported by check_continuity() when the
            # assembly was built; profile() runs on every tuner iteration, so
            # re-warning here would bury the one useful message.
            prof = prof.then(nxt, gap=gap * 1e-3, name=self.name,
                             allow_step=True)
        prof.name = self.name

        if self.ends is not None:
            # Retag by position rather than via set_end_conditions: the first and
            # last segments *are* the assembly's end faces after concatenation,
            # and doing it directly also reopens a 'pec' end, which retagging
            # the PMC apertures cannot.
            left, right = self.ends
            prof._segs[0]['name'] = left
            prof._segs[-2]['name'] = right
        return prof

    @property
    def length(self):
        """Overall axial length of the assembled line, in mm."""
        prof = self.profile()
        if prof is None:
            return 0.0
        zs = [z for z, _ in prof.points]
        return (max(zs) - min(zs)) * 1e3

    def _cell_length_m(self):
        """Mean element length, for the dispersion light line."""
        prof = self.profile()
        if prof is None or not self.n_cells:
            return None
        zs = [z for z, _ in prof.points]
        return (max(zs) - min(zs)) / self.n_cells

    # -- model plumbing ------------------------------------------------------

    def create(self, n_cells=None, beampipe=None, mode=None):
        """Provision the workspace. Native-only: the assembled wall has no gmsh
        ``.geo`` encoding, so ``profile()`` is the single geometry source."""
        if self.projectDir:
            self.self_dir = os.path.join(self.projectDir, self.name)
            os.makedirs(os.path.join(self.self_dir, 'geometry'), exist_ok=True)
            self._write_geometry_snapshot()

    def rebuild(self, parameters, beampipe=None):
        """A fresh assembly, each element rebuilt from its slice of *parameters*."""
        elements = []
        for label, el in zip(self.labels, self.elements):
            prefix = f'{label}:'
            sub = {}
            for key, value in parameters.items():
                if not key.startswith(prefix):
                    continue
                name = key[len(prefix):]
                # Expand a bare name to the element's own slots, exactly as
                # set_tune_value does. An optimiser names its variables however
                # the user wrote the bounds ('cav:A'), and an elliptical cavity
                # rebuilds from 'A_m'/'A_el'/'A_er' only — so passing the bare
                # name straight through dropped the variable and the search
                # silently evaluated the same geometry every time.
                try:
                    slots = self._slots(el, name)
                except ValueError:
                    slots = [name]
                for slot in slots:
                    sub[slot] = value
            merged = {**(el.parameters or {}), **sub}
            elements.append(el.rebuild(merged))
        return type(self)(elements, gaps=list(self.gaps),
                          ends=(None if self.ends is None
                                else tuple(t.lower() for t in self.ends)),
                          allow_steps=self.allow_steps,
                          name=self.name, color=self.color,
                          plot_label=self.plot_label)

    def _reconstruct_state(self):
        return {
            'model': type(self).__name__,
            'module': type(self).__module__,
            'name': self.name,
            'n_cells': int(self.n_cells),
            'beampipe': 'none',
            'parameters': dict(self.parameters),
            'gaps': list(self.gaps),
            'ends': None if self.ends is None else list(self.ends),
            'allow_steps': self.allow_steps,
            'labels': list(self.labels),
            'elements': [el._reconstruct_state() for el in self.elements],
            'dielectrics': [],
        }

    @classmethod
    def _reconstruct_from_state(cls, state):
        elements = []
        for sub in state.get('elements', ()):
            mod = importlib.import_module(sub['module'])
            elements.append(getattr(mod, sub['model'])._reconstruct_from_state(sub))
        ends = state.get('ends')
        asm = cls(elements, gaps=state.get('gaps'),
                  ends=None if ends is None else tuple(t.lower() for t in ends),
                  allow_steps=state.get('allow_steps', False),
                  name=state.get('name', 'assembly'))
        return asm

    # -- composition ---------------------------------------------------------

    def __add__(self, other):
        if not isinstance(other, Cavity):
            return NotImplemented
        rhs = other.elements if isinstance(other, Assembly) else [other]
        return type(self)(self.elements + list(rhs),
                          gaps=list(self.gaps) + [0.0] * len(rhs),
                          ends=(None if self.ends is None
                                else tuple(t.lower() for t in self.ends)),
                          allow_steps=self.allow_steps,
                          name=self.name, color=self.color)

    def __len__(self):
        return len(self.elements)

    def __iter__(self):
        return iter(self.elements)

    def __repr__(self):
        return (f'Assembly({self.name!r}, '
                f'{" + ".join(self.labels)}, {self.length:.4g} mm)')


#: Register the composite with the base class so ``cavity + cavity`` can build one
#: without :mod:`cavsim2d.models.base` importing this module (which would cycle).
Cavity._assembly_class = Assembly
