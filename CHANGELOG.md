# Changelog

All notable changes to cavsim2d are documented here. This project aims to
follow [Semantic Versioning](https://semver.org).

## [Unreleased]

### Added
- Beam-line elements and assemblies. `Beampipe(R, L)` (a drift section, open by
  default), `Bellows(...)` (a corrugated section with analytically validated
  corner radii), `Taper(...)` (a conical transition between two bores, with
  optional rounded corners), and `BLA(...)` (a beam pipe with a lossy absorber
  ring set into its wall, leaving the bore clear) are devices in their own
  right, and any devices concatenate with `+` into an
  `Assembly` — itself a `Cavity`, so it meshes, solves, tunes, optimises and
  takes part in UQ like a single geometry. Joining drops the apertures at each
  junction, so an assembled cavity reproduces the monolithic one exactly;
  `Assembly.set_ends()` controls the two outer faces and overrides whatever the
  end elements declared. Assembly parameters are namespaced `'<label>:<name>'`.
- `Profile.then()`, the geometric concatenation primitive: joins two meridians,
  optionally inserting a drift, and carries each profile's material regions with
  it. A junction whose apertures differ is built as an abrupt radial step and
  reported: `Assembly.check_continuity()` warns once at construction with each
  junction's axial position, radii and step size (and returns them as data), so
  an accidental change of bore is visible without being refused.
  `allow_steps=True` silences it.
- `cav.plot('geometry')` now shades the cavity's material regions, so a beam
  line absorber, a ceramic window or a liner appears in the drawing instead of
  being invisible. `regions=False` suppresses it.
- `Assembly.shade_elements(ax)` shades each element's axial extent behind an
  existing plot, colouring by element kind so two cavities match and a rule marks
  every boundary; `Assembly.element_spans()` returns the same extents as data.
  Band hues are a fixed order (`BAND_COLORS`, from the house palette) chosen by
  maximising the worst OKLab Delta E between *composited* bands under normal and
  simulated colour vision — taking the palette in its natural order left the worst
  pair indistinguishable. Bands are labelled in place by default, since a wash
  that light cannot carry identity by colour alone.
- `cav.plot_axis_field(mode=...)` draws the **signed** on-axis field of any
  solved mode (`cav.axis_field_profile(mode)` returns it as arrays). The cached
  `Ez_0_abs.csv` holds `|Ez|` for the mode of interest only, which cannot show
  how the parts of a multi-cavity line are phased; this can, and it handles the
  complex fields a lossy solve produces.
- Shared corner-rounding helpers (`inscribed_corner`, `corner_offset`,
  `emit_rounded_wall`) in `cavsim2d.geometry.tangency`, used by both the bellows
  convolutions and the taper transitions.
- Higher-order-mode (m-pole) eigenmode API: `eigenmode_config['polarisation']`
  (dipole/quadrupole/… by name or azimuthal number) with per-polarisation
  result folders (`eigenmode/monopole/`, `eigenmode/dipole/`, …) and
  per-polarisation rerun semantics.
- Configurable wall material for eigenmode QOIs: `conductivity` (normal
  conductor, default copper) or `surface_resistance` (fixed Rs, e.g. SRF).
- `MROT` key for wakefield configs (accepts names; `polarisation` kept as a
  deprecated alias) to disambiguate from the eigenmode `polarisation` key.
- `WakefieldSolver` result accessors: `wake_z` / `wake_t` DataFrames and
  `plot_z()` / `plot_t()`.
- Config validation: unknown keys raise a `UserWarning` with a "did you mean"
  suggestion; type/range checks (`processes`, `delta`/`variables` lengths, …)
  raise clear `ValueError`s.
- Runnable example scripts under `examples/` (one per feature) that write
  results and plots to disk.
- Example: `Convergence of every figure of merit`
  (`docs/source/examples/eigenmode/convergence_qois.ipynb`). Sweeps the element
  size and the polynomial order independently and measures every monopole and
  dipole figure of merit against a fine reference, so the integral quantities
  (longitudinal and transverse `R/Q`, `G`, `kcc`, `ff`, the frequency) can be seen
  converging cleanly while the peak-field ratios, which are maxima rather than
  integrals, do not converge at all at low order. This is the check that would have
  caught the peak-field sampling defect.
- Friendly `__repr__` for `Cavity` and `Cavities`.
- Portable `pytest` suite under `tests/`; a `pyproject.toml`-based install.

### Fixed
- Peak surface fields (`Epk`, `Bpk`, and the `Epk/Eacc` / `Bpk/Eacc` ratios) were
  the maximum over the PEC boundary's mesh *vertices*. With order-p elements the
  field along a wall segment is a degree-p polynomial whose maximum generally
  falls between two vertices, so the reported peak was quantised to wherever the
  mesher put a node. In a single run that reads as a small error; under a geometry
  perturbation the mesh is regenerated and the node positions jump, giving
  `Epk/Eacc` about 1% of non-monotonic jitter that does *not* shrink with `h`.
  The wall is now sampled densely inside every boundary element (endpoints
  included), which restores smooth h-convergence and is also faster than the
  per-vertex point search it replaces. Sweeping a geometry variable, the
  quadratic fit to `Epk/Eacc` went from R² = 0.10 to R² = 0.9998.
- Cavities produced by `Cavity.spawn` (sweeps, UQ, optimisation) inherited the
  template's `plot_label`, so a six-point sweep drew six legend entries all
  reading "cavity". They are now labelled by their sweep/row key.
- A tune whose variable does not reach the geometry at all reported the same
  "target may be unreachable" message as one whose variable is merely too weak,
  and pointed at the geometry — sending you the wrong way. The two are now told
  apart by the width of the sampled frequency range, and the no-effect case names
  the usual cause: a shared parameter addressed through a per-cell slot (`Req` is
  one equator radius for the whole cavity and lives in the mid-cell slot, so
  `cell_type={'end-cell': 'Req'}` moves a value nothing reads).
- `cav.tune.plot_convergence()` returned a silent blank after a tune that resolved
  to a design family. A family is a parametric sweep over the split rather than an
  iterative search, so it has no iteration history to draw; the call now says so
  and points at `cav.tune.family`, instead of looking like a failure.
- `show_fields`, `show_mesh` and `show_geometry` with `plotter='matplotlib'`
  ended with an unconditional `plt.show()` and always drew on the current axes,
  so they could not be composed: under the inline backend the show displayed and
  closed the figure, and any later panel of a multi-panel figure landed on a dead
  canvas (plus a stray empty figure). They now take `ax=` and `show=` like the
  `plot_*` methods and return the axes.
- `wakefield_config['mesh_config']` (`DDR` / `DDZ`) never reached ABCI's field
  mesh: the deck writer pinned it at the default while `geo_to_abc` used the
  requested values for contour sampling and beam-pipe length, so sweeping the
  mesh changed the sampling but not the solve and a convergence check came back
  flat.
- Optimisation and UQ over an `Assembly` variable written as a bare element name
  (`'cav:A'`, the way bounds are normally written) silently evaluated the *same*
  geometry for every candidate: `spawn` matched row columns against the parameter
  dict, which holds the per-cell slots `'cav:A_m'`/`'cav:A_el'`/`'cav:A_er'`, so
  the column was dropped. Models now resolve a row column through `_row_slots`,
  which `Assembly` overrides to expand element variables the same way
  `set_tune_value` already did.

- **Monopole surface-loss integral** was mesh-dependent (it sampled `|H|` at
  nodes and wrapped the list in a constant `CF`), making Q/G/Rsh scale with the
  mesh. Now integrates the field over the PEC boundary; the geometric factor G
  is mesh-convergent (~268 Ω for copper).
- Direct-solver backend is no longer hardcoded to PARDISO (Windows-only); it
  falls back to `sparsecholesky` off Windows.
- Cross-platform wakefield launch: clear errors when `ABCI.exe`/`wine` is
  missing instead of a cryptic subprocess failure.
- Optimiser passed float offspring counts to `range()`; `abci_geometry.py` used
  `MROT_DICT` without importing it; the UQ comparison-bar plot raised on
  bookkeeping keys.

### Changed
- The sensitivity-analysis surrogate is now ridge-regularised (`RidgeCV`) rather
  than plain least squares. A degree-2 expansion over the 19 independent inputs of
  a welded 9-cell cavity has 210 coefficients against a few hundred samples, so the
  unpenalised fit reported a flattering in-sample R² alongside a *negative*
  cross-validated one, and the Sobol indices inherited that noise. The penalty is
  chosen by cross-validation, so a figure of merit the polynomial represents exactly
  drives it to zero and fits as before. `surrogate_quality()` gains `alpha`,
  `samples_per_term` and `undersampled` columns, and the parity plot says when a
  poor fit is a sample-count limit.
- `plot_sobol_indices(sort_by=...)` orders the x-axis: `None` (default) keeps the
  sampling order, `'variable'` groups by variable stem (every `Req` before every
  `Ri`, with a rule at each group boundary), and `'S1'`/`'ST'` rank by influence.
  `group=True` remains as the old spelling of `sort_by='variable'`.
- Documentation: the example pages within each category are ordered from the
  simplest to the most involved, the left navigation lists one entry per page
  (section headings no longer flatten into it) and its section captions are
  styled as dividers, notebook result tables pick up the theme's table styling,
  and emphasis is carried by wording rather than bold.
- The wakefield and eigenmode UQ examples draw a mean and its spread as a scatter
  point with an error bar rather than as a bar with `yerr`.
- `CircularWaveguide` is renamed `Beampipe` and its end faces are now `'pmc'`
  (open) by default, so the element composes and the eigenmode
  `boundary_conditions` can steer it. `CircularWaveguide` remains as a
  deprecated alias that keeps the old PEC-closed behaviour.
- Removed/guarded broken legacy API entry points (`Cavities.run_abci`,
  `run_multipacting`, `check_uq_config`, `Cavity.sweep`, multicell UQ) — they
  now raise clear errors instead of failing cryptically.
- Packaging moved to `pyproject.toml`; license metadata corrected to MIT.
- README rewritten around the current API, with a cavity-type capability matrix.

### Known issues
- The m-pole eigensolver currently returns only near-zero gradient-kernel
  eigenvalues (the physical modes are not extracted); m-pole frequencies/QOIs
  are not reliable until this is resolved.
- Multicell UQ and the parameter `sweep` are not available (they raise a clear
  `NotImplementedError`).
- Only `EllipticalCavity` is fully exercised; other cavity types are at varying
  maturity (see the README capability matrix).
