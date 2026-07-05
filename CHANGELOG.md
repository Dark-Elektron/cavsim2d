# Changelog

All notable changes to cavsim2d are documented here. This project aims to
follow [Semantic Versioning](https://semver.org).

## [Unreleased]

### Added
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
- Friendly `__repr__` for `Cavity` and `Cavities`.
- Portable `pytest` suite under `tests/`; a `pyproject.toml`-based install.

### Fixed
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
