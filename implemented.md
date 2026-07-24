# cavsim2d — Implemented (completed-work archive)

This is the **record of work already done** — kept for history and rationale, not a
to-do list. The remaining, still-open items for shipping have been moved to
[SHIPPING_ACTION_PLAN.md](SHIPPING_ACTION_PLAN.md) so that file stays a short focus
list. Any `[ ]`/`[~]` entries that remain below are retained only for the detailed
context they carry; their outstanding portions are what SHIPPING_ACTION_PLAN.md tracks.

Items marked **[verified]** were exercised end-to-end on 2026-07-03 (Windows,
conda env `cavsim2d`, NGSolve 6.2.2506) — re-verify anything you touch afterwards.

Legend: `[ ]` open · `[x]` done · `[~]` partly done · **P0** ship-blocker · **P1** needed
for a good first impression · **P2** soon after release · **P3** nice to have.

---

## What already works and is robust

These flows were run end-to-end and behave correctly — protect them with tests, don't rework them:

- [x] **Eigenmode, monopole** — `cavs.run_eigenmode(config)` / `cav.eigenmode.run()`; results in
  `eigenmode/monopole/`; QOIs, all-modes QOIs, axis field, mesh/field pickles. **[verified]**
- [x] **Eigenmode: ONE unified all-m solver** *(2026-07-12 — replaces the separate monopole and
  m-pole solvers).* The user derived and validated a single product-space formulation
  (`HCurl x H1`, `u_phi = r*E_phi`) that solves every azimuthal order, and it is now the only
  path: `_build_system` / `_solve_system` / `_solve_modes`. `_solve_eigenproblem_mpole`,
  `_build_hcurl_system` and `_solve_hcurl_system` are gone.
  **This fixed a real physics gap:** the old monopole path was HCurl-only, so it had no
  azimuthal unknown and *could not represent monopole TE modes at all* — it silently returned
  TM modes only. At m=0 the two blocks decouple (HCurl -> TM, H1 -> TE), so the monopole
  spectrum now contains the TE modes, correctly reporting `R/Q = 0` (they do not accelerate).
  The key detail is that `fes_phi` is a *separate* `H1(order=p+1, dirichlet="PEC|AXI")` (zero on
  the axis), not the `CreateGradient` space — the latter is used only for the projector's
  freedofs. Verified against the analytic cylindrical-waveguide spectrum (TM *and* TE) to
  **0.000%** for m = 0..3 through the real API, and the TESLA reference QOIs are unchanged
  (TM modes have `u_phi = 0` identically, so the m=0 TM path is numerically the same as before).
  Per-polarisation folders, rerun semantics, and the Panofsky–Wenzel transverse QOI evaluator
  (r0 = aperture/2) are unchanged. The user owns the formulation — see memory
  `mpole-solver-ownership`; integrate what they give you, don't invent weak forms.
  **Consequence to watch:** monopole mode *indices* above the fundamental have shifted (TE modes
  now interleave), so anything keyed on a monopole mode index (dispersion passband grouping,
  `mode_of_interest`) should be re-checked. `CircularWaveguide`'s 1 mm PMC endplate disc was a
  workaround for the old axis handling and can probably now be dropped.
- [x] **Per-polarisation rerun semantics** — `rerun=True` clears only the polarisations in the
  config (dipole rerun never wipes monopole and vice versa, byte-checked); `rerun=False` solves
  only missing polarisations. Legacy flat-layout results still readable via fallback
  (`monopole_dir()` in `cavsim2d/solvers/eigenmode_result.py`). **[verified]**
- [x] **Tuning (elliptical)** — `cavs.run_tune(tune_config)` converges (Req and L stages, hit
  801.58 MHz target); `cav.tuned` reload, `cav.tune.qois`. **[verified]**
- [x] **Eigenmode UQ (simplecell, Stroud3 quadrature)** — runs through `run_eigenmode` with
  `uq_config`; statistics produced (but see P0-1: Q/G/Rsh statistics are polluted by the Ploss
  bug). **[verified]**
- [x] **Wakefield (ABCI, Windows)** — `cavs.run_wakefield(config)`; longitudinal + transversal
  runs read through the solver-agnostic backend: `cav.wakefield.qois` (main-run loss/kick),
  `cav.wakefield.wake_z/wake_t`, `cav.plot('zl')`. **[verified 2026-07-11]**
- [x] **Optimisation** — recently refactored and exercised (`test_optimisation.py`,
  `optimisation.ipynb`); resume support, Pareto plotting suite in `OptimisationSolver`.
- [x] **Multipacting** *(new analysis, 2026-07-16 — ported from the user's PyMultipact)* —
  `cav.multipacting.run(config)`: relativistic electron tracking + secondary emission in the
  cavity's monopole eigenmode field, auto-running the eigenmode solve if needed. New package
  `cavsim2d/analysis/multipacting/` (particles/integrators ported near-verbatim — the user's
  validated physics; sey/fields/driver/metrics/plots the cavsim2d plumbing) + `MultipactingSolver`
  mirroring the eigenmode/wakefield objects. **The crux:** the field adapter reconstructs
  `H = 1j/(mu0 w) curl(E_inplane)` from the product-space in-plane E — cavsim2d drops the `1j`
  (magnitude-only QOIs) but the Lorentz integration needs the physical E-H phase. Descriptive
  plots (`plot_counter`/`plot_final_energy`/`plot_enhanced_counter`/`plot_distance_map`/
  `plot_trajectories`/`plot_sey`); default copper SEY bundled at `data_module/sey/`. Theory page,
  user guide and a baked TESLA example notebook (counter barrier at ~30-40 MV/m). **[verified: the
  counter reproduces the multipacting resonance; `test_multipacting.py` green]**. Parallel sweep
  Parallel sweep workers are plain subprocesses (`python -m ...driver <argsfile>`), NOT
  mp.Process — mp's Windows spawn bootstrap re-executes `__main__` and crashed in notebooks;
  no `if __name__=='__main__'` guard needed anywhere now. Worker stdout → `mworker_{p}.log`,
  failures → `mresults_{p}.err`, both surfaced in the raised error. See memory
  `multipacting-analysis`.
  `pec_maxh` (mm) gives multipacting its OWN surface-refined mesh + monopole field under
  `multipacting/field/` (RestrictH size points, polyline densified to ≤h so straight/tangent
  wall sections refine uniformly — per-edge OCC maxh is stripped by shape-healing). The
  own-field mesh is STRAIGHT (order-1 geometry, order-p FES): the collision polyline must
  coincide with element edges or impacts computed on it fall outside a curved mesh at concave
  wall sections (`Meshpoint not in mesh`); PyMultipact's polygon boundary was exactly this.
  Reuse path (curved eigenmode mesh) is safe for the default equator band only — documented.
  Trajectory viz overhauled: viewer scalar-`set_data` crash fixed (PyMultipact bug), equal
  aspect, counter-peak slider defaults, `color_by='velocity'|'energy'` gradient colouring, and
  `animate_trajectories(epk_i/phi_i/traj, trail, save='.gif'/'.mp4', zoom='auto')` — log colour
  scale mandatory (wall orbits ~1e2 eV vs gap-crossers ~1e5 eV, both physical). Viewer now
  rebuilds the fig INSIDE the ipywidgets callback (inline backend only re-displays figs made
  during the callback — old approach froze on slider moves); `animate_trajectories` returns the
  FuncAnimation AND plays inline in a notebook (`embed='auto'`), with a tqdm frame-render
  progress bar (`progress=True`) on both save and inline. run_sweep prints/stores sweep wall time.
- [x] **All plot_*/show_* default `show=True`** (user request) — display without a manual
  `plt.show()`; `show=False` composes/overlays. `_maybe_show(show)` helper in solver_objects;
  study overlays thread `show=False` into per-cavity calls. See memory `plot-vs-show-api`.
  `cav.eigenmode` results stay byte-identical. Stepwise API: `set_mesh_parameters/set_xrange/
  set_epks/set_phis` (chainable) + `show_mesh`/`show_emission_points` previews; `run()` warns
  (UserWarning) when a config key overwrites a staged value and takes precedence.
  Follow-ups: study-level `study.multipacting` overlay; finer-mesh guidance for production runs.
- [x] **Complete saved configs + kwargs entry** — every analysis has a module-level default config
  (`DEFAULT_TUNE/EIGENMODE/WAKEFIELD/MULTIPACTING_CONFIG` in `solver_objects.py`); `run()` merges
  the user's input over it (`merge_config`, one-level nested) and the **merged** dict is what runs
  and what `config.json` records. Config keys can equally be passed as kwargs (kwargs override the
  dict) on `cav.{eigenmode,wakefield,tune,multipacting}.run` and `study.run_{tune,eigenmode,
  wakefield}`. Consumers must use **truthy** checks (`cfg.get('uq_config')`), never presence
  checks — complete configs carry explicit `None`; fixed in `processes/eigenmode.py`, `study.py`
  (tune/eigenmode/wakefield uq blocks), `models/base.py` (`operating_points`), and
  `config_validation.py` (`n_modes`/`mode_of_interest` treat explicit None as unset — the UQ
  path re-validates the merged config). Restored `TuneSolver.qois` (tune_res.json property,
  dropped in the convergence-frame refactor; test_tune caught it). `mesh_h_metres`
  units guard + staged-overwrite warning use `warnings.warn(UserWarning)` (the verbosity-gated
  `warning()` is silenced by default). Latent quirk left unchanged: top-level wakefield `MT`/`NFS`
  are recorded but only the legacy deck writer consumed them (live writer reads
  `wake_config['MT']`, fallback 4).
- [x] Geometry writing/plotting for elliptical cavities, mesh generation via gmsh→netgen,
  solver-objects API (`cav.eigenmode`, `cav.tune`, `cav.wakefield`, `cavs.optimisation`).

---

## Status after the 2026-07-11 sweep — what is left before v1

**Done since the last sweep (the two seams landed + polish):**
- **All cavity types green.** Every model implements `rebuild()` + declares its own tunable
  handles (`tune_variables()`/`expand_variable()`/`get`/`set_tune_value`), so tuning, UQ and
  optimisation are generic — elliptical, flat-top, pillbox, RF-gun, spline all work. The old
  hardcoded whitelists (`cav.kind` string, `_VALID_BARE_VARS`, substring var matching) are gone.
- **Wakefield solver seam done** (`cavsim2d/solvers/wakefield/`): `WakefieldBackend` protocol +
  `ABCIWakefield`, normalised `WakefieldResult`, `wakefield_config['solver']` registry.
  `get_abci_data`/`abci_data` removed; plot + ZL/ZT objectives read the normalised frames. Main
  run reports `cav.wakefield.qois`; operating points -> `cav.wakefield.qois_op`. A `_DummyWakefield`
  swap test proves solver-independence.
- **Geometry seam** (`Profile`): every type is native; wakefield deck from the profile (arcs kept,
  no NaN); `plot('geometry')` geometry-independent and z-aligned. SplineCavity multicell fixed +
  beampipes + per-cell (mid/end) geometry + arbitrary control-point count.
- **Publication plotting** (`utils/style.py`): warm-cast palette + STIX font. Dispersion: folded
  light line, `bands`, default-on broken axis, per-polarisation colour families.
- Constructor + `run_*` docstrings document every input.

**Also done 2026-07-11 (later):**
- **CI build fixed.** `requirements.txt` was missing `termcolor` + `tqdm` (in `pyproject.toml` but
  drifted) -> `ModuleNotFoundError` on GitHub. Added both; `python-app.yml` and `deploy-book.yml`
  now `pip install .` so `pyproject.toml` stays the canonical dependency source (no more drift).
- **Dispersion phase advances corrected** to `q*pi/(n-1)` (0-mode..pi-mode); see the item below.
- **Import cycle cleared.** Removed the dead `abci_geom = ABCIGeometry()` from `processes/uq.py`
  (the last forcer; the wakefield.py one was already gone) — nothing constructs `ABCIGeometry` at
  import time now.
- **Dead code removed** (git-recoverable): `cavity_legacy.py` (9484 lines) and `quicktools.py`
  (358) — neither imported live.
- **`cavity/` folder tidied** to just the `Cavities` container + façade: the banner moved to
  `utils/welcome.py`, operating-point tables to `data_module/operating_points.py`, the Dakota
  driver to `analysis/uq/dakota.py`; all re-exported so the public API is unchanged.

**Remaining before v1:**
- [~] **Dispersion phase-advance rigour (P1).** *(2026-07-11: switched to the correct convention.)*
  The per-cell phase advances are now `mu_q = q*pi/(n-1)`, `q=0..n-1` — the standing-wave passband
  from the 0-mode (`mu=0`) to the pi-mode (`mu=pi`), via `Cavity._phase_advances(n)`. This fixed
  the old `j*pi/n` labels that omitted the 0-mode and mislabelled the low end. **Still a convention**
  (assigns modes by frequency order, pi-mode at the high-frequency end; mode->passband grouping is a
  frequency heuristic). Fully general values would come from each mode's **on-axis Ez node count**,
  which is monopole-specific (dipole/m-pole Ez vanishes on axis) and needs per-mode field loading —
  left as a refinement.
- [x] **Notebook refresh (P1-6).** *(DONE 2026-07-17.)* Replaced the stale scattered notebooks
  with a clean, **topic-organised** example set under `docs/source/examples/<topic>/`, each
  authored against the VERIFIED current API and **executed with outputs** (nb_execution_mode is
  'off', so committed outputs are what render):
  `eigenmode/elliptical_tesla` + `eigenmode/pillbox`, `tuning/tune_to_frequency`,
  `wakefield/impedance` (ABCI), `multipacting/tesla`, `optimisation/pareto`, `uq/uncertainty`.
  All 7 execute error-free and the Examples toctree (`index.rst`) + each analysis page's
  "worked example" link point at them; a real `sphinx-build` renders all 7. Deleted the stale
  `docs/notebooks/*.ipynb` (broken `from cavsim2d.cavity import`, hardcoded paths, not in any
  toctree) and `playground/test_user_ease.ipynb` — recoverable from git. Fixed the README
  quickstart's broken `from cavsim2d.cavity import` → `from cavsim2d import`.
  **Follow-ups:** (a) `docs/source/examples/multipacting/tesla.ipynb` is ~24 MB — the inline
  animation embeds base64 frames; slim it (fewer frames / lower dpi, or reference a saved gif)
  before it bloats every clone. (b) The `examples/*.py` scripts (`01_eigenmode.py`…) still carry
  stale calls (`from cavsim2d.cavity import`, `plot_fields`, `get_abci_data`, `plot('zl')`) — the
  new notebooks supersede them for tutorials, but the README still points at the scripts; either
  refresh the scripts or repoint the README at the notebooks.
- [x] **Eigenmode axis-field bug + QOI unification + doc-nav polish (2026-07-17).**
  - **Real bug fixed:** `evaluate_qois` wrote `Ez_0_abs.csv` whenever `save_dir` was set, so the
    all-modes loop clobbered it with the LAST mode — a 1-cell fundamental's `plot_axis_field`
    showed a higher mode (node at centre, peak in the beam pipe). Gated behind `write_axis`
    (primary mode-of-interest only); m-pole `Ez_r0_abs_mode_N.csv` was per-mode so unaffected.
  - **QOI evaluators UNIFIED (user request):** one `evaluate_qois(…, m)` for every azimuthal
    order; `evaluate_qois_mpole` deleted. Only voltage-derived quantities branch on m (on-axis vs
    off-axis Panofsky–Wenzel, 2π vs π azimuth); **P_loss is now the H1-projection integral for all
    m** (the deferred P2-2 accuracy upgrade — re-baselined m≥1 P_loss/Q/G). Verified: monopole
    TESLA QOIs byte-identical, dipole pillbox still matches the analytic closed form
    (test_eigenmode 31 + test_uq 8 + test_multipacting 11 green). Solve DRIVERS stay split by
    design (`_solve_eigenproblem` monopole+AMR vs `_solve_modes`), both on the unified
    `_build_system`. See memory `mpole-solver-ownership`.
  - **Optimisation quiet:** tune-accuracy failures (`'could not be reached'`) are expected noise
    (candidates that can't tune are discarded) — added to the EA's `suppress_errors`.
  - **`plot_compare_fm_scatter`/`_bar`(uq=True) fixed:** UQ metric keys are polarisation-prefixed
    (`monopole:freq [MHz]`) since the objective-naming change, but the plots matched/labelled on
    the bare name → empty/KeyError. Now strip the prefix. These are the built-in QOI-spread plots
    the UQ example uses.
  - **Docs:** examples nested by topic (per-topic `index.rst` + toctree), `globaltoc_collapse:
    False` (fixes sidebar sections vanishing off-page), UQ split into eigenmode + tune notebooks
    (+ wakefield-UQ note), `qois_df` trimmed to 7 columns; sphinx build clean (all example
    cross-refs resolve).
- [ ] **Docs site (P2-5).**
- [x] **API polish + WEPB015 multicell-UQ/SA replication (2026-07-18).** Large user-review pass:
  (1) removed the deprecated `Cavities` alias — `Study` everywhere (code/tests/docs/notebooks).
  (2) `plot_compare_*` → namespaced `plot_*` on both single cavity and study
  (`cav.eigenmode.plot_fm_scatter/bar`, `study.wakefield.plot_hom_*`/`plot_power_*`,
  `.plot_compare(kind=)`, cross-domain `Study.plot_all_*`); impl bodies stay private on Study,
  namespaces are thin `show`-aware delegators, single-cavity builds a transient one-cavity Study.
  (3) `cav.plot()` now also defaults `show=True`; removed redundant `plt.show()` from every example.
  (4) `plot_spectrum` REDESIGNED — polarisation-coloured spectral comb + passband cluster-mean
  triangles + optional `bands=True` (all solved polarisations, not the monopole stem).
  (5) `plot_tm010_spectrum(reference=, kde=True)` — default probability-of-resonance distribution
  (Corno Fig 5), `kde=False` freq-vs-1-based-mode-index; overlays Corno et al. 2020 RI/EZ data.
  (6) Boxed/raised legends in `house_style` (frameon + shadow).
  (7) **Impedance now sums ALL m-pole modes** — transverse spans every `m>=1` multipole
  (dipole+quadrupole+…), longitudinal unchanged (off-type R/Q=0); see [[impedance-and-hom-extension]].
  (8) **WEPB015 example** `examples/eigenmode/multicell_uq_sa.ipynb`: multicell MC UQ with
  `independent_half_cells` (perturb each half-cell, weld seams → before/after-continuity, Fig 3/4),
  `plot_fm_scatter(uq=True)` (Fig 5), a **clean Sobol' SA path** (`analysis/uq/sobol_sa.py`:
  polynomial surrogate + SALib Saltelli/Sobol', `run_sensitivity`/`plot_sobol_indices` +
  `surrogate_quality`, Fig 6), and the TM010 passband vs Corno — all from the 9-cell UQ cavity.
  Full suite 193 passed; new SA + welding tests added. **Forward work** (user): extend to HOMs
  (eigenmode config takes n_modes+polarisations into the UQ; tune targets monopole fundamental,
  tune-var stays `L`; the paper's ~0 freq-std comes from per-sample tuning — `uq_config` already
  supports nested `tune_config`).
- [x] **Tune-in-UQ observability + refinements (2026-07-19).** Per-half-cell quarter tune now:
  folder `multicell_tune/` with each `halfcell_<i>/` a **full cavity object**
  (`geometry/` geodata.geo+mesh.step+mesh.pkl, `eigenmode/` gfu_EH.pkl+qois.json, `tune/`
  tune_info.json w/ convergence+timing + tune_log.txt), root `tune_log.txt` (per-cell + TOTAL);
  **assembled UQ-point geometry saved** (`<sample>/geometry/`: contour.csv, half_cells.csv,
  parameters.json, mesh.pkl — native-only assembly had no `.geo`); **tune mode selection**
  `tune_config['mode']='monopole:4'` (default pi-mode) threaded into `cavity_quarter(m=,mode_index=)`;
  eigenmode-cfg inherits parent minus uq_config; **quarter beampipe `L_bp=2*L`** (was 4*L);
  **~34% faster tuning** via default quarter `mesh_h=12` (over-resolved at 20; freq identical to 3 s.f.;
  overridable via `tune_config['mesh_h'|'mesh_p']`). **Per-analysis run logs** — `utils/run_log.py`
  `RunTimer` writes a single `<analysis>/run_log.json` (steps + total_s) for eigenmode, wakefield,
  multipacting. `plot_spectra(break_axis=True)` → N independent per-band panels via subplot_mosaic
  (per-panel autoscaled y). api_guards/uq/plot_compare/eigenmode suites green.
- [ ] **Multicell ABCI writer for wakefield UQ (TODO, feasible).** Wakefield UQ on an
  independently-tuned multicell is currently `NotImplementedError`, but the ABCI writer already loops
  over cells writing each one — so a new `write_cavity_geometry_cli_multicell`-style ABCI deck writer
  (per-half-cell geometry from `half_cells()`, same beampipe convention) would unblock gathering
  wakefield/impedance QOIs per UQ point alongside the eigenmode QOIs. (User confirmed it *can* be
  written; my earlier "ABCI can't express it" was wrong.)
- [x] **Generalise optimisation to non-elliptical types.** *(DONE 2026-07-17.)* The real gap was
  **not** in `optimisation.py`'s live path (already generic via `_resolve_suffixed_var` + `spawn`)
  but in `OptimisationSolver.run` (`solver_objects.py`), which **always fabricated an
  `EllipticalCavity` template from the bounds**, ignoring the cavity added to the study. The
  pillbox test passed only by accident (`Ri`/`L`/`Req` exist on elliptical too — a false positive).
  Fix: the template is now `study.cavities_list[0].rebuild(...)` (a copy of the added cavity, so
  its type drives resolution/spawn/rebuild); the bounds-derived elliptical is only the fallback
  when no cavity was added (bare `Cavities(dir).run_optimisation(cfg)`). Second bug found on the
  way: `_evaluate_generation` read the tuned value as `last['parameters'][var]`, which `KeyError`s
  for a spline (`'p3_r'` is control point `'p3'[r]`, not a flat key) — now via the model's
  `get_tune_value`. Dead elliptical-only `run_uq` deleted; `generate_initial_population` no longer
  injects `alpha_i/alpha_o` zeros or the `Req`+1 magic (both elliptical-only; ignored elsewhere).
  Verified end-to-end: **pillbox, RF-gun (tunes `R6`), and spline (tunes/sweeps control-point
  coords `p3_r`/`p2_r`/`p4_r`) all optimise**. `test_optimisation.py`: elliptical fallback + pillbox
  + a **discriminating spline test** (its `p3_r` doesn't exist on elliptical, so a template
  regression raises loudly). See memory `optimisation-generalised`.
- [x] **Advanced examples: "UQ everywhere" + robust optimisation fixed (2026-07-18).**
  New `docs/source/examples/advanced/` section (an `index.rst` preamble —
  *deterministic until you add a* `uq_config` — plus a toctree). Three executed
  notebooks spanning cavity types to stress the model-agnostic machinery:
  `eigenmode_uq.ipynb` (spline `p3_r` + RF-gun `R6`, QOI spread via the built-in
  `plot_compare_fm_scatter(uq=True)`), `robust_tuning.ipynb` (flat-top, robust
  tune of `A,B`, reads `cav.tune.folder/'uq_tune_results.json'`),
  `robust_optimisation.ipynb` (elliptical, `uq_config` nested in the opt's
  `eigenmode_config`, ranks candidates by `E[obj]+6·std[obj]`, shows `E[]`/`std[]`
  columns + `opt.plot_pareto`). The old `examples/uq/` folder was folded into
  `advanced/`; `index.rst` toctree + `uq.rst` link repointed; sphinx build clean.
  **Robust optimisation was actually broken** — the UQ block in
  `_evaluate_generation` (`analysis/optimisation.py` ~L653) had three bugs, all
  fixed: (1) iterated `df['key']` but `key` is the DataFrame **index** →
  `for key in df.index`; (2) resolved `uq.json` from the wrong dir → now via the
  cavity object (`Path(scav.tuned.self_dir)/'uq'/'uq.json'`, fall back to
  `scav.self_dir`, skip if neither exists); (3) matched prefixed objective names
  (`monopole:Epk/Eacc []`) against `EIGENMODE_QOIS` which holds **bare** names →
  `_bare(o[1])`. Also `df.merge(on='key')` → `df.join` (key is the index).
  Guarded by `test_robust_optimisation_ranks_by_uq_objective` (asserts the
  `E[`/`std[` columns appear); full `test_optimisation` + `test_api_guards` green
  (12 passed).
- [x] **Wakefield UQ fixed + now a live example (2026-07-18).** Three bugs closed
  the last "UQ everywhere" gap (every analysis except multipacting now takes a
  `uq_config`): (1) `processes/uq.py` wrote the result to `cav.wakefield_dir/uq.json`
  — a directory that need not exist → FileNotFoundError; now writes `cav.uq_dir/uq.json`
  (the same file the eigenmode branch writes and the optimiser + display read).
  (2) The study-level reader `get_wakefield_qois` (`study.py`) read from the stale
  `<projectDir>/wakefield/ABCI/<name>/uq.json` layout → repointed to `cav.uq_dir`.
  (3) `Cavities.run_wakefield` truncated a `ZL`/`ZT` objective `['min','ZL',[f…]]`
  to `['', 'min', 'ZL']` — dropping the frequency windows AND shifting the QOI
  name into the goal slot, so every objective came back empty; now keeps the full
  `[goal, name, args]` form (as the optimiser's `objectives_unprocessed` already
  did). Also passed the configured `solver` into `get_wakefield_objectives_value`
  so a non-ABCI backend's frames are readable, and deleted two stray
  `print('it is here…')` debug lines in `study.py`. Verified with a canned-backend
  smoke test AND a **real ABCI** study run (two `ZL` windows, mean±std). New
  `docs/source/examples/advanced/wakefield_uq.ipynb` (executed, 1 table + 1 figure)
  wired into the `advanced/` toctree; the "wakefield has a bug" caveat removed from
  `advanced/index.rst`; `wakefield.rst` links the UQ variant. `test_wakefield` +
  `test_uq` + `test_api_guards` green (29 passed).
- [x] **API polish + plot redesigns + WEPB015 multicell-UQ/SA replication (2026-07-18).**
  A six-part sweep from a user review:
  - **`Cavities` removed.** It was only a back-compat alias for `Study` (no users
    yet); deleted the alias + export, replaced every usage across code / tests /
    notebooks / README with `Study`.
  - **`plot_compare_*` → namespaced `plot_*`.** The 10 `Study.plot_compare_*`
    methods became `study.eigenmode.plot_fm_scatter/bar` / `plot_compare`,
    `study.wakefield.plot_hom_*`/`plot_power_*`/`plot_compare`, and cross-domain
    `Study.plot_all_scatter/bar`. Also available on **single cavities**
    (`cav.eigenmode.plot_fm_scatter(uq=True)` = one cavity's UQ mean±std + design
    point) via a transient one-cavity Study. Impl bodies stay private on Study;
    the public methods are thin `show`-aware delegators. No aliases.
  - **`show=True` everywhere.** `cav.plot(what, show=True)` now shows too (body →
    `_plot_dispatch`, wrapper adds `_maybe_show`); the study overlays pass
    `show=False`. Removed the redundant `plt.show()` from every example notebook
    (kept only after hand-built `plt.subplots` figures).
  - **`plot_spectrum` redesigned** — a spectral comb (`vlines`) coloured by
    polarisation with passband cluster-mean triangles and optional `bands=True`
    shading; shows ALL solved polarisations (was monopole-only stem).
  - **`plot_tm010_spectrum(reference=)`** — overlays the fundamental monopole
    passband on reference spectra (mean±std); the example overlays Corno et al.
    2020 (NIM A 971, 164135) RI+EZ data.
  - **WEPB015 replication** (Udongwo & van Rienen, IPAC'25) as
    `docs/source/examples/eigenmode/multicell_uq_sa.ipynb`: independent-half-cell
    multicell UQ that **welds** (averages) the shared seams
    (`perturb_half_cells_independent`, gated by `uq_config['independent_half_cells']`,
    writes `nodes_before_continuity.csv`), Monte-Carlo distributions
    (`plot_fm_distribution` Fig 3, `plot_node_distribution` Fig 4), mean±std
    (`plot_fm_scatter` Fig 5), and a **clean variance-based Sobol' SA path** —
    new `cavsim2d/analysis/uq/sobol_sa.py` (polynomial surrogate + SALib
    Saltelli/Sobol', replacing the broken legacy LHS-misusing `Sobol`), exposed as
    `cav.eigenmode.run_sensitivity()` / `plot_sobol_indices()` (Fig 6). Verified
    physics (freq↦Req, peak-fields/R/Q↦Ri) + welding narrows the shared-DOF
    spread. SA deps **SALib + scikit-learn** are already declared (pyproject.toml
    + requirements.txt). Tests added:
    `test_sobol_indices_finds_dominant_variable`,
    `test_perturb_half_cells_independent_welds_seams`. All notebooks re-executed
    with outputs; refs (Corno2020, Udongwo2025) added to `refs.bib`.
- [x] **Review round on the above (2026-07-18):**
  - **TM010 plot is now the distribution.** `plot_tm010_spectrum` gained
    `kde=True` (default) — the probability-of-resonance distribution (each mode a
    Gaussian of its spread, Corno Fig 5 style; this cavity's modes as vlines);
    `kde=False` keeps the freq-vs-mode-index view, now **1-based**.
  - **Boxed/raised legends** — house style sets `legend.frameon/​fancybox/​
    shadow/​edgecolor/​facecolor`; dropped every `frameon=False` in plot code.
  - **Impedance sums ALL m-pole modes** (`EigenmodeSolver.impedance`) — each mode
    via its type-appropriate R/Q (off-type = 0). Longitudinal unchanged, transverse
    now spans every `m>=1` multipole, not just dipole. `test_impedance` green.
  - **SA surrogate goodness-of-fit** (user asked to *see* how well the surrogate
    fits): `sobol_sa.analyse()` returns `(sobol, surrogate)`; `run_sensitivity`
    also writes `uq/surrogate.json`; new `cav.eigenmode.surrogate_quality()`
    (R²/CV-R²/RMSE/max-err DataFrame) + `plot_surrogate_quality()` (CV parity
    plots). Flags unreliable indices (small-sample peak-field ratios fit poorly,
    like the paper's geometry factor). `test_surrogate_quality_reported` added.
    WEPB015 notebook gained the surrogate section; all example notebooks
    re-executed for the new legend/impedance/TM010 outputs.
- [x] **README support matrix** — already all-green for UQ/Opt across the five types, and that is
  now **true and test-backed** (it was aspirational while optimisation was elliptical-only). Gun
  `Epk/Eacc` mesh-dependence (¹) and spline coordinate-handle (²) caveats remain, correctly noted.
- [~] **Docs site (P2-5) content.** *(Rendered example notebooks landed 2026-07-12.)*
  `myst_nb` was already a Sphinx extension but had no execution config and no notebook in the
  toctree. Added:
  - **`docs/source/examples/eigenmode_pillbox.ipynb`** — a live, fully-rendered eigenmode
    tutorial built only on the public API (geometry -> `eigenmode.run` -> mesh + fundamental
    field -> `qois_df` -> 100 modes x m=0..8 vs the closed form -> R/Q showing the monopole TE
    modes at exactly zero). Executed and committed **with outputs** (5 figures, 918 modes).
  - `conf.py`: `nb_execution_mode = 'off'` + dollarmath/amsmath. Notebooks are **not** re-executed
    at build time — that needs NGSolve + gmsh on the runner and ~10 min for the 900-mode solve.
    Re-run manually after library changes (command is in a comment in `conf.py`); flip to
    `'cache'` to let Sphinx execute.
  - `index.rst` gained an **Examples** toctree; `eigenmode.rst` documents `qois_df`.
  - Fixed a stale `automodule:: cavsim2d.cavity` in `cavsim2d.rst` left by the `cavity/` removal.
  Verified with a real `sphinx-build`: page renders, all 5 figures present, no cavity warning.
  **Update 2026-07-17:** the old `docs/notebooks/*.ipynb` are DELETED and replaced by the
  topic-organised `docs/source/examples/<topic>/` set (see P1-6 above) — 7 executed notebooks now
  in the Examples toctree. **Still open:**
  **`.github/workflows/deploy-book.yml` runs `jupyter-book build` but there is no `_config.yml`/
  `_toc.yml`** — the docs are a Sphinx project (`docs/source/conf.py`, `sphinx_immaterial`), so
  that workflow cannot be building what it claims to. Point it at `sphinx-build`.
- [ ] **Deeper stale-code sweep (optional).** The legacy ABCI deck writer chain
  (`geometry/writers/abci.py` `ABCIGeometry` + `analysis/wakefield/abci_code.py`) is superseded by
  the native `profile` -> `_write_abc` path but still wired through `utils/shared_functions.py`;
  the `Dakota` driver (`analysis/uq/dakota.py` + `analysis/uq/dakota_scripts/`) has a hardcoded
  `D:/Dropbox/...` path and is non-functional. Both are entangled / in the public API — decide
  keep-vs-remove deliberately rather than delete blind.
- [~] **Standalone analysis + `Study` rename + restructure** *(A/B/most-of-C done 2026-07-11.)*
  A cavity is now analysable without a `Study`: `cav.set_workspace(folder)` / lazy
  `_ensure_workspace()` (each solver `run()` provisions `./<name>/` in the CWD if none set) +
  `cav.save(folder)` / `cav.load(folder)`. The manager was **renamed `Cavities` -> `Study`**
  (`Cavities` kept as an alias; both exported) and its file **moved out of `cavity/` to top-level
  `cavsim2d/study.py`** — `from cavsim2d import Study`. **`cavity/` folder removed 2026-07-11:**
  `cavsim2d/__init__.py` now imports every public name from its real home (`models`, `study`,
  `analysis.uq.dakota`, `data_module.operating_points`, `utils.welcome`); all `from
  cavsim2d.cavity import ...` sites (package + tests) were repointed to `from cavsim2d import ...`.
  `optimisation.py` moved to `analysis/` and **generalised to all cavity types** (spawn column map
  now via the model's `_resolve_suffixed_var`, not a hardcoded `A..Req`). Verified: `import
  cavsim2d` clean, `Cavities is Study`, 157 tests collect, `test_api_guards`/`test_plotting_style`
  green; two stray mathtext `SyntaxWarning`s in `utils/sensitivity.py` fixed in passing.
  **Remaining (Part C,
  riskiest):** split the 3,900-line `study.py` into `study.py` + a `compare_plots.py` mixin (the
  ~1,500-line `plot_compare_*`/`plot_cavities_*`/`plot_*_fields` suite), and **merge `processes/`
  into `analysis/`** — do the merge last, one topic at a time, suite green between each.

  **Proposed target layout** (direction for the Part-C merge; public API stays at top level):

  ```
  cavsim2d/
    __init__.py        # public surface: Study, Cavity + subclasses, apply_style, ...
    constants.py
    study.py           # Study manager (+ ComparePlotsMixin from compare_plots.py)
    models/            # device models: Cavity, Elliptical*, Spline, RFGun, Pillbox, ...
    geometry/          # Profile, primitives, tangency, plotting, beampipes, writers/
    analysis/          # ALL analysis — one subpackage per topic, driver + algorithm together:
      eigenmode/       #   parallel driver + NGSolve solver (from processes/ + solvers/NGSolve)
      tune/            #   driver + pyTuner            (from processes/ + analysis/tune)
      wakefield/       #   driver + backends (abci, future ngsolve) + readers
      uq/              #   driver + dakota
      optimisation.py  #   EA
    data_module/       # result readers (abci_data), operating_points
    utils/             # style, printing, welcome, shapes, config_validation, ...
  ```

  *(`cavity/` façade removed 2026-07-11 — the top-level package is the only public surface.)*

  This collapses the confusing three-way `processes/` (drivers) + `solvers/` (NGSolve/ABCI) +
  `analysis/` (algorithms) split into one `analysis/<topic>/` per topic. `solvers/solver_objects.py`
  (the `cav.eigenmode`/`.tune`/`.wakefield` accessors) stays as the user-facing accessor layer.
  Fold the dead legacy ABCI writer (`geometry/writers/abci.py` + `analysis/wakefield/abci_code.py`)
  and non-functional `analysis/uq/dakota*` cleanup into this pass.
- Documented non-blockers (leave with clear messages): `Cavity.sweep` (`NotImplementedError` +
  workaround), multicell *wakefield* UQ (`NotImplementedError`; simplecell works). The in-house
  NGSolve wakefield solver is future work — the seam is ready, ABCI covers v1 on Windows.

---

## P0 — Correctness bugs and ship-blockers

- [x] **"All cavity types green" programme.** *(DONE 2026-07-11; see the sweep summary above.)*
  Historical context — the matrix below is how things stood on 2026-07-10, before the two seams
  landed. **The README matrix was wrong in both directions** — measured, not read:

  | | Eigenmode | Tuning | Wakefield | UQ / Opt |
  |---|---|---|---|---|
  | `EllipticalCavity` | OK | OK | OK | OK |
  | `Pillbox` | OK | OK | OK (needs `L_bp > 0`) | **TypeError in spawn** |
  | `RFGun` | OK | **silent no-op** | fails | **TypeError in spawn** |
  | `EllipticalCavityFlatTop` | OK *(was broken)* | **silent no-op** | fails (no `.geo`) | fails |
  | `SplineCavity` | OK *(was broken)* | **silent no-op** | fails | **TypeError in spawn** |

  Better than advertised: flat-top and spline eigenmode are OK (native `Profile`), not the README's
  warning symbol — indeed the flat-top's `write_geometry` never wrote a file, so its ⚠️ was
  generous. Worse than advertised: the `➖` cells are not "unavailable", they are **silent no-ops**.
  `cavs.run_tune(...)` on a flat-top returned normally, printed a red `ERROR::`, and left
  `cav.tuned = None`; every non-elliptical UQ dies with
  `TypeError: Pillbox.__init__() got an unexpected keyword argument 'name'`.

  **Everything non-green reduces to two missing seams, not to a lack of modularity.**

  1. *"Rebuild a cavity of this type from a parameter dict."* Needed by `clone_for_tuning` (tuning)
     and `spawn` (UQ/optimisation). Elliptical and Pillbox implement it ad hoc; the other three
     don't, and the generic `spawn` passes `name=` / `geo_filepath=` kwargs their constructors
     reject. One `rebuild(parameters)` hook per model would let the base implement both, once.
  2. *"Give me this cavity's contour as points."* Needed by wakefield. `geo_to_abc` gets it by
     **regex-parsing the `.geo` text**, understanding only `Point(...)` and `Ellipse(...)`. So the
     flat top fails (no `.geo`), the spline fails (its wall is a `BSpline`/`Bezier` the regex cannot
     see — it would silently emit the *control polygon*), and half-cell multicell cavities fail
     (`geo_filepath is None`). `Profile` already is this seam.

  **Three incidental defects fixed (2026-07-10)** — none architectural:
  - `pyTuner.py:86` gated suffixing on `cav.kind == 'elliptical cavity'`; the flat-top's kind is
    `'elliptical cavity flat top'`, so `Req` was never expanded to `Req_m` -> `KeyError('Req')`.
    Replaced with a `Cavity.uses_cell_suffixes` capability (True for the two elliptical families).
  - A `≈` (U+2248) in a tuner message raised `UnicodeEncodeError` on the Windows cp1252 console —
    a tuning *failure* crashed instead of reporting itself. `printing._printable()` now degrades
    unencodable characters for every print site; the message itself is ASCII.
  - `EllipticalCavityFlatTop.create()` took `tune=` while every other model takes `mode=`, so the
    tuner's `mode='tune'` raised `TypeError`, which `tune_function` caught and reported as
    **"degenerate geometry"** — sending the root finder wandering and blaming the user's
    parameters. Renamed; the flat top now tunes on the full geometry (it has no quarter-cell
    writer), as the pillbox already does.

  **And the swallows that hid them.** `processes/tune.py` caught bare `Exception`, printed, and
  returned; `pyTuner.tune_function` caught bare `Exception` and set `degenerate = True`. A new
  `TUNE_DEFECTS` tuple (`NotImplementedError, TypeError, AttributeError, KeyError, ImportError,
  UnicodeError, IndexError`) now propagates through both, so "this type cannot tune" is a loud,
  accurate error rather than a silent no-op. Genuine degenerate candidates are still tolerated —
  `test_optimisation` still passes.

  **Decisions taken with the user (2026-07-10):**
  - Add beampipe options to `RFGun` and `SplineCavity`; **default beampipe length = 3x the axial
    length of the device**. (ABCI refuses to run without beam pipes at both ends of at least 5 mesh
    lengths — this is an ABCI requirement, not a cavsim2d one.)
  - Build the **full wakefield seam**: `Profile`-driven geometry, a `WakefieldSolver` protocol
    (`run` / `read_results`), and a normalised `wakefield/<pol>/qois.json` schema. Move ABCI file
    reading behind it and retire `cav.get_abci_data()` / `cav.abci_data` in favour of
    `cav.wakefield.qois`, so swapping ABCI for the planned in-house NGSolve time-domain solver is
    one new class plus a registry entry.
  - Order: incidental bugs (done) -> seam 1 (`rebuild`) -> seam 2 (`Profile` -> wakefield).

  **Wakefield seam DONE (2026-07-11).** `cavsim2d/solvers/wakefield/` is the backend protocol:
  `base.py` (`WakefieldBackend` with `run`/`read_dir`/`read`/`write_qois`/`read_qois`,
  `WakefieldResult`), `abci.py` (`ABCIWakefield` wrapping `ABCI.solve` + `ABCIData`),
  `__init__.py` (`get_backend`/`register_backend`/`BACKENDS`). Backend chosen by
  `wakefield_config['solver']` (default `'abci'`), persisted to `wakefield/config.json` by the
  worker (`processes/wakefield.py` `_run_abci`) so reads resolve the same backend. Normalised
  frames (`wake_z`/`wake_t`, with `Re(Z)`/`Im(Z)` for the objectives) and per-pol
  `wakefield/<pol>/qois.json`. `WakefieldSolver` accessor exposes `.backend/.result/.wake_z/
  .wake_t/.qois`. **`get_abci_data`/`abci_data` removed**; `plot('zl'/'zt'/'wpl'/'wpt')` reads
  the frames via `_wake_xy`; ZL/ZT objectives (`get_wakefield_objectives_value(..., solver=)`)
  compute from the frames. Dead `calc_k_loss`/`all` and the unused `ABCI`/`ABCIData`/
  `ABCIGeometry` imports removed from `processes/wakefield.py` (drops one of the two sites forcing
  the `geometry/writers/abci` import cycle; the other is `processes/uq.py`). Verified: ABCI k_loss
  parity unchanged; a `_DummyWakefield` swap test drives run + `cav.wakefield.qois` + `plot('zl')`
  + a ZL objective with **no ABCI**. To add the NGSolve solver: subclass `WakefieldBackend`,
  `register_backend(it)`, pass `solver='ngsolve'`.

  **Seam 1 done (2026-07-10).** Every model now implements one hook,
  `rebuild(parameters, beampipe=None)` -> a fresh bare instance. The base class implements
  `clone_for_tuning` and `_load_tuned_from_disk` on top of it once, and `spawn` was rewritten on the
  same hook (it used to call `type(self)(name=..., geo_filepath=...)`, which no model's constructor
  accepts — hence `TypeError: Pillbox.__init__() got an unexpected keyword argument 'name'` for every
  non-elliptical UQ). Per-type `clone_for_tuning` / `_load_tuned_from_disk` duplicates deleted from
  `elliptical.py`, `pillbox.py`, `circular_waveguide.py` (the waveguide's passed 3 args to a 2-arg
  constructor and could never have run). `write_geometry` signatures normalised to
  `(parameters, n_cells, beampipe, write=...)` so the generic clone can call any of them.

  Result — measured:
  - **Flat-top tuning works**: `Req_m` 171.20 -> 169.29, tuned frequency 790.0000 MHz against a
    790 MHz target (previously a silent no-op).
  - **UQ works for Pillbox, FlatTop and RFGun**: e.g. pillbox freq E = 1153.05 MHz (sd 0.134),
    flat top 780.16 (sd 0.017), gun 212.63 (sd 0.206).
  - `EllipticalCavity.spawn` keeps its override (it pre-aligns `Req` across cells to avoid a
    spurious warning); everything else uses the generic one.

  Tests: `test_tune.py` (rebuild is the single hook; round-trips every model; flat top now tunes;
  the base hook names itself) and `test_uq.py::test_uq_works_for_every_cavity_type`.

  **Seam 2 done (2026-07-10).** The wakefield contour now comes from `Profile`, not from
  regex-parsing the `.geo` text. `Profile.contour_points(ds)` densifies the wall (validated against
  the analytic quarter-ellipse perimeter to 2.5e-7 at ds = 0.1 mm);
  `cavsim2d/geometry/beampipes.py` adds `abci_shape()` / `abci_contour()` / `ensure_beampipes()`.

  **ABCI always needs beam pipes** ("at least 5 mesh lengths" at *each* end) — its requirement, not
  the cavity's. Ends whose pipe is missing get one of **3x the device's axial length** (override with
  `wakefield_config['beampipe_length']`, metres). Ends that already have one are untouched. The RF
  gun already has a 150 mm downstream drift tube, so only its cathode side gets a pipe.

  **A polyline is not enough.** Densifying an elliptical cavity's iris arc — which meets the beam
  pipe *tangentially* — makes ABCI's mesher emit **NaN wake potentials** (43,165 NaN tokens in
  `cavity.top`; `LONGITUDINAL WAKE MIN/MAX = NaN` in `cavity.out`). Deterministic, and independent
  of point spacing (NaN at ds = 20, 6, 2, 1.25 mm; clean at 10 and 4 mm) and of snapping the deck to
  the ABCI mesh. Pillbox (straight lines), spline and gun (arcs not tangent to a pipe) were always
  clean. So `abci_shape()` keeps ABCI's native `-3.` arc primitive for `arc` and `ellipse` segments
  and densifies only splines.

  Verified: the Profile deck reproduces the legacy `.geo` deck **exactly** — pillbox k_loss
  -0.310700 both ways, elliptical `bp=both` -0.180300 both ways — so no existing physics changed.
  Every type now runs NaN-free: elliptical `bp=none` -0.180200 (vs -0.180300 for `bp=both`, which it
  must match, since it is the same cavity with different pipe lengths — a good check on the pipe
  insertion), flat top -0.200400, spline -0.157400, gun -0.309200. `contour_ds` and `beampipe_length`
  registered in `WAKEFIELD_KEYS`.

  Guard update: `test_abci_abort_raises_instead_of_silently_producing_nothing` used a pillbox with
  `L_bp = 0`, which now runs (pipes added). It drives `_raise_if_abci_aborted` directly instead, and
  a new test asserts the pillbox case now succeeds.

  **Fidelity question: RESOLVED, no action needed.** I previously recorded that ABCI has no ellipse
  primitive and that `-3.` describes a circle, making the iris wall wrong. That was **wrong**. ABCI's
  `-3.` card takes exactly two following lines — centre, then end — and denotes an *axis-aligned
  elliptic arc* (the start is the previous point). Two points on the ellipse give two linear equations
  in `1/a^2`, `1/b^2`, so the arc is exact. Recovered from the real TESLA deck: iris card ->
  `semi_z = 12.0000`, `semi_r = 19.0000` mm; equator card -> `42.0000 / 42.0000` mm — the design
  values. `abci_shape()` already passes `seg['center']`, so it is exact. No biarc work.
  Only constraint: ABCI cannot express a *rotated* ellipse; ours are axis-aligned by construction.

- [x] **Generalise the tune variable resolver (was blocking the Tuning column).**
  `_VALID_BARE_VARS` was a hardcoded elliptical-only whitelist. Replaced by model-owned hooks:
  `Cavity.tune_variables()`, `expand_variable()`, `get_tune_value()`/`set_tune_value()`, derived from
  each model's own `self.parameters`. `SplineCavity` overrides them (control points `[z, r]` -> the
  handles are coordinates, `'p3_r'`). Adding a geometry now needs no edit to any central list; an
  unparameterised (imported CAD/mesh) geometry says so instead of reporting an unknown name.

  Four further bugs of the same species, found and fixed on the way:
  - `_resolve_suffixed_var` split on `'_'`, so the pillbox's `L_bp` silently resolved to `L`.
  - UQ expanded variables by substring (`if k_var in parameter_key`): spline `'p3_r'` matched
    nothing -> `k = 0` random variables -> `ZeroDivisionError` in the Stroud3 quadrature.
  - `_load_tuned_from_disk` did `float(v)` under `except (TypeError, ValueError): pass`, silently
    **dropping** a tuned spline control point.
  - `RFGun.get_eigenmode_qois(self)` narrowed the base `(self, config=None)` -> UQ raised TypeError.

  Verified: RFGun tunes `R6` (60 -> 100.6 mm, 210.0000 MHz); SplineCavity tunes `p3_r` and the
  re-solved tuned geometry independently returns 1488.02 MHz; UQ runs for flat-top, pillbox, gun
  and spline. README matrix now all green.

  **Still open:** `ABCIGeometry` / `analysis/wakefield/abci_code.py` is dead in the live path —
  `abci_geom = ABCIGeometry()` is built at import in `processes/wakefield.py` and `processes/uq.py`
  but only ever *called* from `cavity_legacy.py`. It is what forces the `geometry/writers/abci.py`
  import cycle. Deleting those two lines should let the cycle exemption go.

- [x] **P0-1: Fix the monopole surface-loss integral (wrong physics, mesh-dependent Q/G/Rsh).**
  `evaluate_qois` in `cavsim2d/solvers/NGSolve/eigen_ngsolve.py` computed
  `Ploss = Integrate(y * InnerProduct(CF(Hsurf), CF(Conj(Hsurf))), ..., definedon='PEC')`.
  `CF(list_of_point_values)` builds a *constant vector* CF, so it evaluated
  `sum(|H_i|^2) * ∫y dl` — scaling with the number of surface nodes, i.e. with the mesh.
  **Fixed 2026-07-03**: project the azimuthal H = curl(E)/(mu0 w) into a continuous H1 field
  (NGSolve cannot SIMD-evaluate curl(GridFunction) directly on a boundary) and integrate its
  trace over the PEC boundary. **[verified mesh-convergent]**: G stable at 266.5/267.5/268.5 Ω
  across a 10x element-count change (was 2109/147/179 Ω before) — physically correct for
  copper. **Still TODO: re-baseline** every stored/published Q, G, Rsh, GR/Q number and re-run
  the UQ examples, since all of these shift with the corrected Ploss. (Note: the m-pole
  evaluator's segment-wise Ploss was already mesh-convergent; upgrading it to the same
  H1-projection method for higher accuracy is folded into P2-2.)
- [x] **P0-2: Commit the working tree.** *(Done 2026-07-03.)* ~25 modified files (m-pole solver, monopole folder
  restore, per-pol rerun, reader updates, notebooks) are uncommitted. Commit in logical chunks
  (m-pole feature / folder-layout restore / reader fixes) so this state is recoverable.
- [~] **P0-3: Remove or port the broken legacy API layer.** *(Partly done 2026-07-03.)*
  DONE: deleted dead `Cavities.run_abci` / `run_multipacting` / `check_uq_config`; `Cavity.sweep`
  (and the `Cavities.sweep` wrapper it drives) now raise a clear `NotImplementedError` with a
  workaround instead of crashing in the removed legacy solver; multicell UQ
  (`uq_config['cell_complexity']='multicell'`) now raises a clear `NotImplementedError` at both
  the eigenmode and wakefield call sites instead of a cryptic TypeError deep in a worker.
  Working paths (eigenmode mono+m-pole, simplecell UQ, tune, wakefield) re-verified green.
  STILL TODO:
  - Properly **port `sweep`** to the current pipeline (update `self.parameters`, invalidate the
    geometry snapshot, re-run via `self.eigenmode.run()` per point) — it was only guarded, not
    restored. `sweep.ipynb` depends on it.
  - Properly **port or delete multicell UQ** (`uq_parallel_multicell` + `uq_multicell_s` in
    uq.py; `NGSolveMEVP.cavity_multicell` with its stale `_solve_eigenproblem` signature) — only
    guarded so far.
  - Decide on the remaining legacy internals still present but now largely unreachable:
    `Cavity.run_eigenmode` / `Cavity._run_ngsolve` (base.py) and `Cavity.run_wakefield` /
    `_run_abci`; `Cavity._run_ngsolve` is still called by `Pillbox.run_eigenmode`, so fold this
    into the pillbox work (P2-4) rather than deleting blind.
  - pyTuner.py dead paths (~315–395, ~740–773) calling `ngsolve_mevp.cavity_quarter` / `.cavity`
    / `.createFolder`.
  Original notes below still apply to the leftover items:
  longer exist (`ngsolve_mevp.createFolder/.cavity/.cavity_flattop/.cavity_quarter`) or use old
  function signatures — they crash the moment a user finds them:
  - `Cavity.run_eigenmode()` and `Cavity._run_ngsolve()` (base.py ~646–840) — also calls
    `uq_parallel` with the pre-refactor 5-argument signature.
  - `Cavity.sweep()` (base.py ~554) — **the sweep feature is broken** because it calls the
    legacy `run_eigenmode`; `sweep.ipynb` advertises it. Port sweep to
    `cav.eigenmode.run()` / `run_eigenmode_s`.
  - `Cavity.run_wakefield()` legacy method (base.py ~686) — verify against the new
    `processes/wakefield.py` path; keep exactly one implementation.
  - `Cavities.run_abci()` / `Cavities.run_multipacting()` (cavities.py ~2643) — call
    `cav.run_abci`/`cav.run_multipacting`, which don't exist. Delete or implement.
  - `Cavities.check_uq_config()` — calls `cav.check_uq_config`, which doesn't exist.
  - `pyTuner.py` ~315–395 and ~740–773 (`cavity_quarter`, `ngsolve_mevp.cavity`,
    `createFolder`) — dead tune/UQ code paths.
  - `NGSolveMEVP.cavity_multicell()` — calls `self._solve_eigenproblem(run_save_directory,
    mesh, mesh_p)` with a stale signature (missing `cav`); paired with `uq_multicell_s`
    (uq.py ~561) which is the only caller path. Fix the signature and the flat qois layout it
    writes/reads, or clearly mark multicell-UQ unsupported for v1.
  Decision to make once: **one way to do each thing** — the solver-object API
  (`cav.eigenmode.run()`, `cavs.run_eigenmode()`) is the keeper; delete the rest rather than
  deprecate, since nothing external depends on it yet.
- [x] **P0-4: Platform guard for wakefield.** *(Done 2026-07-03.)* Added `run_abci_exe()` /
  `abci_exe_path()` in `solvers/ABCI/abci.py`: checks the executable exists (clear
  `FileNotFoundError` pointing at the README if not), launches directly on Windows, and via
  `wine` on other platforms (clear `EnvironmentError` if wine is absent). Both the live path
  (`ABCI.run_abci`) and the `abci_geometry.py` launch sites now use it; wakefield smoke
  re-verified on Windows. **STILL TODO (non-blocking):** decide the ABCI redistribution story —
  the repo currently *bundles* `ABCI.exe` while the README says to download it; confirm KEK
  terms allow bundling or drop it from the repo/wheel and keep the download instructions.
- [x] **P0-5: License metadata mismatch.** *(Done 2026-07-03.)* `setup.py` now sets
  `license='MIT'` and the `MIT License` classifier (was LGPL + full licence text); also fixed
  the malformed duplicate `Topic ::` classifier and added `long_description_content_type`.
  Fuller packaging cleanup remains under P1-1.

## P1 — Needed for a good first release

- [x] **P1-1: Packaging overhaul.** *(Done 2026-07-03.)*
  - Replaced `setup.py` with `pyproject.toml`; version is now `0.1.0` (was the non-PEP-440
    `13.08.2024`), `requires-python = ">=3.10"` (was `>=3.0`). ABCI.exe shipped via
    `package-data`.
  - Reconciled dependencies into one canonical list in `pyproject.toml` (core deps + `jupyter`
    and `dev` extras); `requirements.txt` is now explicitly the dev/docs superset. scikit-learn
    and SALib are core (the package import chain pulls them in). Verified `pip install -e .`
    builds and imports cleanly and reports version `0.1.0`.
  - Made the `IPython` import in `welcome.py` lazy (and the logo read graceful) so the package
    imports headlessly and in a wheel where `docs/` isn't shipped.
  - **NOTE:** revisit whether scikit-learn/SALib *should* be core — they are only pulled in via
    `utils/shared_classes.py`; decoupling that would let them move to an `analysis` extra
    (ties into P3-1 experimental-module cleanup).
- [x] **P1-2: README rewrite.** *(Done 2026-07-03.)* Rewritten around the current API and
  folder layout (`<project>/cavities/<name>/eigenmode/<pol>/…`): fixed badges (were
  CavityDesignHub), install (`pip install -e .`, Python ≥3.10, extras), the ABCI path
  (`cavsim2d/solvers/ABCI/`) + wine note, `Cavity`→`EllipticalCavity`, and the `cell complexity`
  footgun→`cell_complexity`. Added a runnable quickstart (verified: TESLA 1-cell → 1300.2 MHz,
  R/Q 113.5, G 271.1), an m-pole eigenmode section (`polarisation`/`n_modes`, per-pol rerun,
  Panofsky–Wenzel QOI note), the `MROT` wakefield key with the eigenmode-vs-wakefield
  disambiguation, config-validation note, a pointer to `examples/`, and a "running the tests"
  section. Every code snippet uses a verified API pattern; quickstart + m-pole snippets run
  verbatim.
- [x] **P1-3: Config validation with helpful errors.** *(Done; type/range checks completed by
  user 2026-07-05.)* `cavsim2d/utils/config_validation.py` has per-config validators
  (eigenmode/tune/wakefield/uq/optimisation) wired into the `Cavities.run_*` entry points.
  Unknown keys raise a standard `UserWarning` with a difflib "did you mean" suggestion; nested
  sub-configs validated; the `'cell complexity'`→`'cell_complexity'` footgun flagged. Type/range
  checks now raise `ValueError` via the `require(cond, msg)` helper: `processes` positive int,
  `n_modes`/`nmodes` positive int, `MT`/`NFS` int, and `delta`/`epsilon` length == `variables`
  length. `'nmodes'` accepted as an alias. Covered by `tests/test_config_validation.py`.
- [x] **P1-4: Resolve the `'polarisation'` terminology collision.** *(Done 2026-07-03.)*
  Wakefield now uses `'MROT'` as the canonical key; `'polarisation'` is a deprecated alias (a
  one-line `info` note on use). Both accept 0/1/2 or names ('monopole'/'longitudinal',
  'dipole'/'transverse', 'both') via `resolve_mrot()` in `solvers/ABCI/abci.py`, used at every
  read site (`ABCI.solve`, `geo_to_abc`, `Cavities.run_wakefield`) so `cav.wakefield.run()` and
  the `Cavities` path behave the same. Also made `get_abci_data` load only the polarisations
  that were actually run, so a single-mode run (`MROT='dipole'`) no longer crashes looking for
  the other; verified end-to-end (dipole-only writes just `transversal/`, `wake_t` populated,
  `wake_z` gracefully empty) and the default both-mode path still passes.
- [x] **P1-5: Portable test suite + green CI.** *(Done 2026-07-03.)* Replaced the hardcoded-path
  scripts with a real pytest suite under `tests/` (all writing to `tmp_path`), sharing fixtures
  via `conftest.py`. **16 tests pass locally** in the `cavsim2d` env:
  - `test_eigenmode.py` — monopole folder layout, physical QOIs, P0-1 mesh-independent G,
    m-pole pillbox vs analytic, per-pol rerun isolation, legacy flat-layout fallback.
  - `test_tune.py` — tune hits target frequency. `test_uq.py` — simplecell UQ statistics +
    multicell-UQ-guard raises. `test_wakefield.py` — Windows/ABCI-gated ABCI smoke.
  - `test_api_guards.py` — dead wrappers stay removed, sweep raises. `test_optimisation.py` —
    small end-to-end run (moved in from repo root).
  Deleted obsolete `test_flat_structure.py` / `test_readme.py` / `verify_tuner_fix.py`.
  Solver tests `importorskip('ngsolve'|'gmsh')` so CI stays green where those aren't installed;
  wakefield gated on Windows+ABCI. Added `.flake8` (stable excludes incl. the dead legacy
  files) and simplified the CI workflow; whole-repo critical lint (E9/F63/F7/F82) is clean.
  Also fixed two real bugs found while writing the tests: `optimisation.py` passed float
  offspring counts to `range()` (added `int()` guards in `crossover`/`mutation`), and
  `abci_geometry.py` used `MROT_DICT` without importing it (latent `F821`).
  **STILL TODO:** the eigenmode solver hardcodes `inverse="pardiso"` /
  `Preconditioner(..., inverse="pardiso")` in `_solve_eigenproblem` — PARDISO may be absent in
  the pip `ngsolve` build on Linux/macOS, so CI-on-Ubuntu (and non-Windows users) could hit a
  runtime failure rather than a clean skip. Make the direct-solver backend configurable with a
  fallback (e.g. `sparsecholesky`/`umfpack`). Tracked also under cross-platform support.
- [ ] **P1-6: Notebook refresh.** Every notebook has hardcoded personal paths; most still use
  the removed `cavs.save(...)` API and old `SimulationData` layout (convergence_study, misc,
  sweep, rf_gun, pillbox, readme_test, optimisation partially). Update to current API with
  relative/temp paths, re-execute top-to-bottom, and strip stale outputs. These are the de
  facto tutorials — they must run cleanly on a fresh install. (eigenmode.ipynb m-pole section
  is current; tune_test and wakefield_analysis are close.)
- [x] **P1-7: Implement `WakefieldSolver` result accessors.** *(Done 2026-07-03.)* `wake_z` /
  `wake_t` now load ABCI output into DataFrames (impedance spectrum `f [MHz]`, `|Z|` + wake
  potential `s [m]`, `W`); `plot_z()` / `plot_t()` plot either `quantity='impedance'` or
  `'wake'`. Verified against real ABCI output (peak |Z_L| at ~809 MHz, at the cavity
  fundamental). Empty frames / clear "run first" message when no output exists.
- [~] **P1-8: Repo hygiene.** *(Partly done 2026-07-03; more 2026-07-09.)* Removed tracked dead
  file `tmp_shared_functions.py` (and its now-stale `.flake8` exclude). The rest (`thesis/`,
  `artefacts/`, `_build/`, `.idea/`, `__pycache__/`, `*.egg-info/`) are already gitignored and
  untracked.

  2026-07-09: deleted 396 lines of unreachable geometry code
  (`write_cavity_geometry_cli_quarter`, `write_pillbox_geometry`, `arcToTheta`). Kept the
  Multipac exporters and **wired them into the public API** as
  `EllipticalCavity.export_multipac()` with a test — writing that test immediately caught a
  latent bug (its `file_path` is a *file*, not a directory, contrary to how it read). Added
  explicit imports to the 8 modules that were silently relying on names leaking out of
  `from ... import *` chains (`Path`, `re`, `json`, `np`, `pd`, `os`, `math`); `shared_functions`
  now carries a `per-file-ignores` entry since its star re-exports are deliberate.

  **Convention: imports live at the top of the file.** Hoisted 51 function-level imports down to
  7 (2026-07-09), and moved the top-level imports that were stranded below code in `data_utils.py`
  up into the header (deduping a second `import numpy`/`import re`). The 7 survivors are all
  deliberate and each carries a `# Deferred:` comment giving the reason — either an optional heavy
  dependency (`netgen`, `ngsolve`, `IPython`) or a genuine import cycle
  (`base` <-> `cavities`, `uq` <-> `wakefield`, `solver_objects` <-> `tune`/`optimisation`/
  `models.elliptical`/`cavities`). `test_api_guards.py::test_imports_live_at_module_top` enforces
  this: it fails on any nested import lacking a `# Deferred:` rationale, and on any top-level
  import sitting below the first `def`/`class`.

  STILL TODO: `cavity_legacy.py` (9,484 lines) is gitignored — a local-only reference copy, so it
  won't ship via git — but it still sits in the `cavsim2d/` package dir, so a `setuptools`
  **wheel build would include it**. Its content is recoverable from history (it is `cavsim2d/cavity.py`
  at `be38758`), so deleting the local copy is safe; alternatively exclude it from the wheel
  (`py-modules`/MANIFEST). Left in place pending a decision — it is untracked, so removal is
  the user's call.

## P2 — Shortly after (quality and coverage)

- [x] **Multicell representation: independently varying cells.** *(Done 2026-07-09 — geometry
  layer and eigenmode UQ.)* Goal: run UQ over the **whole cavity** — every half-cell an
  independent random variable — instead of over the mid-cell and end-cell groups.

  **Canonical representation (new):** `EllipticalCavity.half_cells()` returns a `(2*n_cells, 7)`
  array in mm. Row `2k` is the forward (left) half of cell `k+1`, row `2k+1` its backward (right)
  half, each with its own `(A, B, a, b, Ri, L, Req)`. `set_half_cells(arr)` installs an explicit,
  independently varying set (pass `None` to revert to the parameter-derived, uniform-mid-cell
  geometry). `continuity_violations()` enforces the physics: **`Req` is shared by the two halves
  of a cell; `Ri` is shared across the iris between adjacent cells** — so a middle cell's iris
  cannot move without moving both halves that meet at it. `Cavity.half_cells()` raises
  `NotImplementedError` for pillbox/gun/waveguide/spline, which have no half-cell decomposition.

  `cavsim2d/geometry/contours.py` gained `half_cell_sequence()` and
  `elliptical_profile_from_half_cells()`; `elliptical_profile()` is now a thin wrapper that expands
  mid/end_l/end_r into half-cells. Verified behaviour-preserving: for uniform mid-cells the new
  path reproduces the old contour to 1e-15 on every point, across 1/2/3 cells and all beampipe
  configurations. Verified live: on a 3-cell TESLA, narrowing only the two middle irises 35 -> 33 mm
  moves the eigenmode 1296.418 -> 1290.504 MHz and R/Q 332.95 -> 355.35; widening only cell 2's
  equator gives 1291.410 MHz.

  **Eigenmode multicell UQ now works** (2026-07-09). `uq_parallel` branches internally on
  `uq_config['cell_complexity'] == 'multicell'`; the `NotImplementedError` guard is gone.

  - `perturb_half_cells(cav, config)` (in `utils/shapes.py`) perturbs the half-cell array directly.
  - `half_cell_free_variables(n_cells, variables)` derives the **free** random variables from the
    constraints, so continuity holds *by construction* — no post-hoc projection. A shared entry is
    one variable whose delta is applied to every row it spans:
    `Req` -> `n_cells` vars (one per cell), `Ri` -> `n_cells + 1` vars (two apertures plus
    `n_cells - 1` irises), everything else -> `2 * n_cells` vars. Verified for n = 1, 2, 3.
  - `half_cells_to_dataframe()` writes `nodes.csv` with columns `A1, B1, ..., Req1, A2, ...`, one
    index per half-cell — which is the scheme `shapes_to_dataframe`'s docstring always claimed and
    `enforce_continuity_df` expects.
  - `EllipticalCavity.spawn_half_cells()` builds the perturbed-cavity container.

  Validated on a 2-cell TESLA (`variables=['Ri','Req']`, stroud3): 10 samples x 28 columns, and
  `uq.json` gives freq E = 1294.360 MHz (sd 3.004), R/Q E = 221.004 (sd 9.091). The simplecell run
  of the same cavity gives 12 samples x 21 columns, freq sd 4.113, R/Q sd 1.457 — the much larger
  R/Q spread under multicell is expected, since independently varying cells break field flatness.

  **No-silent-fallback guard.** `write_geometry` can only emit a uniform-mid-cell contour, so a
  cavity with independent cells is native-only. `set_half_cells()` therefore clears `geo_filepath`,
  `create()` writes no `.geo`, and `NGSolveMEVP._build_mesh` raises if it has neither a profile nor
  a `.geo` — rather than quietly solving a *different* cavity. `set_half_cells(None)` reverts and
  restores the gmsh path.

  **Dead code removed** (net -424 lines): `uq_parallel_multicell` and its worker `uq_multicell_s`
  (pre-refactor signatures, drove `cavity_multicell`), and `Cavity._run_ngsolve` (never called — the
  live one is `processes.eigenmode._run_ngsolve` — and it referenced an undefined bare `OC_R`).
  `enforce_continuity_df` lost a stray `print(m)` debug; its regex still admits `_m`-style suffixes
  that the following `int(m.group(2))` would raise `ValueError` on.

  STILL OPEN:
  - **Multicell wakefield UQ** is unimplemented and still raises a clear `NotImplementedError`
    (ABCI needs the flat geometry array, which the native path does not produce).
  - `cavity_multicell`, `write_geometry_multicell` and `write_cavity_geometry_cli_multicell` are now
    unreachable. They should retire together with the two legacy encodings (`shape_multicell['IC']`
    of shape `(8, n_cells-1, 2)`, and the flat `16*n_cells` array), leaving `half_cells()` canonical.
  - `to_multicell()` still runs on every `create()` and populates `shape_multicell`; it is now
    redundant.

- [x] **P2-1: Tuning for non-elliptical cavities — Pillbox now tunes.** *(Done 2026-07-05.)*
  The tuner core already handled unsuffixed params (it only suffixes when
  `kind=='elliptical cavity'`); the blockers were per-type plumbing. Added
  `Pillbox.clone_for_tuning` + `Pillbox._load_tuned_from_disk`, and fixed
  `_filter_params_for_ct` (it returned an empty dict for unsuffixed params, so the
  tuned-parameter snapshot was blank). Verified end-to-end: `Req` 100→105.3 mm,
  freq hits 1100.0 MHz, `cav.tuned` is a Pillbox that re-solves to target; elliptical
  tuning unaffected. README matrix updated (Pillbox tuning ✅). Test in
  `test_cavity_types.py::test_pillbox_tuning`. STILL TODO (smaller): flattop/spline/
  RF-gun `clone_for_tuning` if those flows are wanted.
  *(Prior partial note, superseded:)*
  Tuning stays elliptical-only: the tuner is coupled to the elliptical suffixed-parameter model
  (`_resolve_suffixed_var('Req','mid-cell')→'Req_m'`, cell-type stages), while Pillbox uses
  unsuffixed params (`Req`, `L`, `Ri`). Attempting to tune a non-elliptical cavity now raises a
  clear, actionable message ("Frequency tuning is currently only supported for EllipticalCavity;
  <Type> tuning is not implemented …") instead of a bare `clone_for_tuning` NotImplementedError.
  Documented in the README capability matrix (tuning ➖ for non-elliptical). STILL TODO (larger
  refactor): generalise the tuner to single-cell unsuffixed params + add per-type
  `clone_for_tuning` to actually enable pillbox/flattop/spline tuning.
  Original note: `clone_for_tuning` is implemented only by
  `EllipticalCavity`; flattop/pillbox/spline/RF-gun raise `NotImplementedError` via the base
  class (pillbox still carries its own dead `run_tune` mentioning SLANS). Either implement per
  type or document clearly which features apply to which cavity type (a capability matrix in
  the README).
  *(Documented 2026-07-04:)* Added a capability matrix to the README from actual smoke tests —
  `EllipticalCavity` full (verified), `Pillbox` eigenmode verified (1159.6 MHz ≈ analytic
  TM010), `EllipticalCavityFlatTop`/`SplineCavity`/`RFGun` marked not-verified (flattop eigenmode
  failed to write geometry with a plain elliptical param list — needs the correct flattop param
  format). STILL TODO: implement `clone_for_tuning` per type (or keep documented), and verify
  flattop/spline/RF-gun end-to-end (ties into P2-4).
- [x] **P2-2: m-pole integration depth — m-pole QOIs are now UQ/optimisation objectives.**
  *(Completed 2026-07-10.)*

  The blocker was ambiguity, not plumbing: monopole and dipole `qois.json` share **18 identical
  key names** (`freq [MHz]`, `R/Q [Ohm]`, `G [Ohm]`, `Q []`, `Epk/Eacc []`, ...), so a bare
  `'R/Q [Ohm]'` objective cannot say which polarisation it means. New module
  `cavsim2d/solvers/objectives.py` defines the grammar:

  ```text
  monopole:R/Q [Ohm]              # the polarisation's primary mode of interest
  dipole:R/Q_t [Ohm/m^(2(m-1))]   # a dipole-only QOI
  dipole:2:freq [MHz]             # dipole, mode 2 (1-based)
  1:freq [MHz]                    # azimuthal number instead of a name
  ```

  Separator `:` (no QOI name contains one); only the first two fields are split, so the QOI may
  contain anything. **Bare names are rejected** (breaking, deliberate): `'freq [MHz]'` used to
  silently mean the monopole's. Naming a polarisation in an objective is enough — `uq_parallel`
  unions it into `eigenmode_config['polarisation']` automatically.

  Wired into `uq_parallel` (the QOI table is now built from the objectives, columns named by
  `Objective.column`) and `Optimisation` (reader rewritten). Validated at config time by
  `_check_objectives` in both `validate_uq_config` and `validate_optimisation_config`; wakefield
  objectives (`ZL`/`ZT`) carry no polarisation and pass through untouched.

  **Three bugs fixed on the way.**
  1. `uq.py` filtered objectives against a hardcoded whitelist and **silently dropped** anything
     not on it — so a typo (`'R/Q [Ohms]'`) or any dipole-only QOI vanished, and the statistics
     were quietly computed over whichever objectives happened to match. Now
     `read_objective_values` raises, listing the available QOIs for that polarisation.
  2. `Optimisation` built its objective row from `qois.items()` order rather than the objective
     list order, so values could **misalign with their names**. Now ordered by the objectives.
  3. `optimisation.py` then wrote those values into DataFrame columns chosen by a *second*
     hardcoded whitelist. Replaced with the objective list itself.

  Dead code removed: `uq(proc_cavs_dict, ...)` (no callers; it held the silent-drop whitelist).

  Verified end to end on a 2-cell TESLA: a single UQ run with `['monopole:freq [MHz]',
  'monopole:R/Q [Ohm]', 'dipole:freq [MHz]', 'dipole:R/Q_t [Ohm/m^(2(m-1))]']` produces
  monopole 1293.73 MHz and dipole 1660.97 MHz (the deflecting passband sits above the monopole
  one, as it must), and `dipole:1:freq` equals `dipole:freq` since the primary m-pole mode
  defaults to 1. Tests in `test_objectives.py` (new), `test_config_validation.py`.

  Note the remaining `monopole_dir(...)` call sites (`pyTuner.py`, `wakefield.py`,
  `solver_objects.py`, `base.py`) are genuinely monopole-specific — the tuner tunes the monopole
  frequency, and HOM power reads the monopole fundamental — so they are correct as they stand.

- [x] **P2-2 (earlier phase): m-pole dispersion + docs.** *(2026-07-05.)*
  DONE: `plot_dispersion(pol='dipole'|'quadrupole'|…)` now plots the m-pole passband (reads
  `eigenmode/<pol>/qois_all_modes.json`; monopole slices `[1:n_cells+1]`, m-pole `[0:n_cells]`
  since it has no DC mode) — verified a 2-cell dipole passband sits above the monopole one;
  `Cavities.plot_dispersion(pol=...)` overlays all cavities. Documented in the README m-pole
  section: the headline `qois.json` index convention (m-pole = lowest mode 0 vs monopole =
  π-mode index `n_cells`), and that `kcc` for m≥1 reuses the monopole formula (limited meaning
  for a deflecting passband). Test `test_mpole_dispersion_plot`.
  STILL TODO (larger, design-level): make dipole/quadrupole QOIs usable as UQ / optimisation
  objectives — needs a polarisation-qualified objective naming scheme and for the
  UQ/optimisation readers to pull from `eigenmode/<pol>/qois.json` instead of the monopole
  folder. Documented as a known limitation in the README.
- [x] **P2-3: Material/physics constants configurable.** *(Done 2026-07-04.)* Added a
  `surface_resistance(w, conductivity, rs)` helper and `SIGMA_COPPER` constant; `evaluate_qois`
  and `evaluate_qois_mpole` take `conductivity` / `surface_resistance_ohm`, threaded from
  `eigenmode_config['conductivity']` (normal conductor, default copper) or
  `['surface_resistance']` (fixed Rs, e.g. SRF). Verified: G is material-independent (271.1 Ω
  across copper/SRF), Q scales inversely with Rs, SRF niobium (Rs=1e-8) gives Q≈2.7e10.
  Documented in the README eigenmode table; covered by two tests. (Was: copper 5.96e7 hardcoded.)
  STILL TODO (related): `beta=1` is assumed; the `Eacc` normalisation-length convention
  (`L_m`, active length = 2·L·n_cells) should be documented and overridable.
- [~] **P2-4: RF-gun and pillbox flows.** *(Mostly done 2026-07-04.)* Verified eigenmode
  end-to-end via `cavs.run_eigenmode()`: **Pillbox** → 1159.6 MHz (≈ analytic TM010),
  **RFGun** → 213.1 MHz (credible VHF gun). Removed the Pillbox legacy `run_tune`/`run_eigenmode`/
  `run_wakefield`/`_run_ngsolve`/`_run_abci` overrides (dead, interactive `input()`, old
  `Cavities/` layout) so it inherits the working base methods. Reworked base
  `Cavity.run_eigenmode`/`run_wakefield` to delegate to the modern solver-object pipeline
  (`self.eigenmode.run()` / `self.wakefield.run()`) — so the per-cavity `cav.run_eigenmode()`
  now works for **every** cavity type (also closes the P0-3 leftover). Tests in
  `test_cavity_types.py`.

  *(Closed 2026-07-09.)* RF-gun `Eacc` normalisation: `L_norm` falls back to the on-axis field
  extent when the cavity has no `L_m`, with an explicit `eigenmode_config['normalization_length']`
  override — covered by `test_rfgun_qois_normalised_sensibly`. Flattop and spline now run natively
  (see the geometry entry). **Pillbox wakefield works end to end**, and closing it exposed two bugs:

  1. **ABCI aborts silently.** It exits 0 even when it refuses to run, writing `*** STOP ***` into
     `cavity.out` and leaving `cavity.pot`/`cavity.top` empty. `run_abci_exe` now checks for that
     and raises `RuntimeError` carrying ABCI's own reason. Trigger: a pillbox declaring
     `beampipe='both'` with `L_bp = 0` gives zero-length pipes — "THE BEAM PIPES AT BOTH ENDS ARE
     TOO SHORT. THEY MUST HAVE AT LEAST 5 MESH LENGTH."
  2. **`test_pillbox_wakefield` was false-green.** It ran `beampipe='none', L_bp=0`, which ABCI
     refuses, and asserted only `os.path.exists(cavity.top)` — which a 0-byte file satisfies. Now it
     uses `L_bp = 50 mm` and asserts `getsize(...) > 0`. Note ABCI *requires* beam pipes at both
     ends, so a `beampipe='none'` cavity cannot be wakefield-simulated at all.
- [ ] **P2-5: Docs site.** `docs/` is a MyST/Jupyter-Book site with a deploy workflow — rebuild
  after P1-6, check `deploy-book.yml` still deploys, add an API-reference page generated from
  docstrings, and a "concepts" page (project folder layout, config dictionaries, QOI
  definitions incl. m-pole conventions).
- [x] **P2-6: Uncommitted-state UX.** *(Done 2026-07-05.)* `run_eigenmode_s` now deletes the
  stale UQ artefacts (`eigenmode/uq*.json` and the `<cav>/uq/` perturbation folder) when the
  monopole is rerun (`rerun=True`, `0 in pols`), since UQ is derived from the monopole solve —
  so old UQ can't outlive the eigenmode results it came from. Test
  `test_uq.py::test_rerun_clears_stale_uq_artefacts`. NOTE (separate latent bug, not blocking):
  `Cavity.get_uq_fm_results` reads `eigenmode/uq.json` but the UQ writer saves to `<cav>/uq/uq.json`
  — the container path works, but the single-cavity `cav.run_eigenmode(uq_config=...)` read path
  is looking in the wrong place.

- **Index-convention fixes (2026-07-05, downstream of the user's solver refactor).** The monopole
  solver now filters the spurious gradient/DC mode (`mask_ = evals_ > 1`), so index 0 is the
  fundamental (was index 1). Updated all downstream conventions: headline `mode_idx = n_cells-1`
  (`solve` + `cavity_multicell`), `kcc` reference → index 0 (monopole + m-pole), `plot_dispersion`
  passband slice → `[0:n_cells]` reading each polarisation's own `qois_all_modes.json`. Also made
  the UQ aggregation numeric-only (the unified qois schema added a non-numeric `'polarisation'`
  key that broke `weighted_mean`), and unbroke `get_eigenmode_qois` callers (`config` optional +
  polarisation normalisation + non-empty guard). Full suite green.
- [x] **`mode_of_interest`: which eigenmode becomes the headline `qois.json`.** *(New, 2026-07-09;
  multi-mode support 2026-07-10.)*
  `eigenmode_config['mode_of_interest']` is **1-based** (mode 1 = lowest of the passband) and may be
  a single int, **a list of any length** (a polarisation may have any number of modes of interest),
  or a dict of either keyed by polarisation name or azimuthal number — each polarisation may carry a
  different count, e.g. `{'monopole': [1, 2, 3, 4], 'dipole': 1}`. The **first**
  mode listed is primary and its QOIs become `qois.json`; **all** of them are written to
  `qois_moi.json`, keyed by 1-based mode. Resolved by `NGSolveMEVP.modes_of_interest(...)`, which
  returns a list of 0-based indices (deduplicated, order preserved).
  Defaults: **monopole -> `n_cells`** (the pi-mode, the
  operating mode of an accelerating structure — for a 1-cell cavity, gun or pillbox this is mode 1,
  so it degrades gracefully to non-accelerator geometries); **m-pole -> 1** (lowest of the
  deflecting passband; an m-pole spectrum has no spurious DC mode to skip). Resolved once, at the
  config boundary, by `NGSolveMEVP.mode_of_interest(cav, m, cfg, n_modes)` -> internal 0-based
  `mode_idx`. Name chosen over `analysis_mode`/`operating_mode`: it names the thing (an eigenmode)
  and its role, without accelerator jargon that reads oddly for a waveguide.

  Recorded in `qois.json` as a **string** (`'3'`), like `'polarisation'` — UQ coerces the QOI table
  to numerics and drops text columns, so a bare int would be "averaged" into a meaningless
  mean/stddev. Validated in `validate_eigenmode_config` (positive int, <= `n_modes` when given, dict
  values checked per polarisation); a typo warns with a "Did you mean 'mode_of_interest'?" hint.
  Verified on a 3-cell TESLA: passband 1279.27 / 1287.64 / 1296.42 MHz, default headline = mode 3 =
  1296.4183 MHz (unchanged from before, so backwards compatible), `mode_of_interest=1` -> 1279.2713,
  `{'monopole': 2}` -> 1287.6418. Tests in `test_eigenmode.py` + `test_config_validation.py`.

- [x] **P2-7: Error handling audit.** *(Done 2026-07-09.)* All **73 asserts** in the package are now
  `require(cond, msg)`, which raises `ValueError`. This was a real bug, not style: 56 of them read
  `assert cond, error(msg)`, and `error()` *prints and returns None*, so the user saw
  `AssertionError: None` — the message never reached them. And `python -O` strips asserts entirely,
  so every one of those checks silently vanished in optimised runs. Both demonstrated before
  changing anything. Seven asserts carried no message at all and got one written.

  Also removed 8 now-duplicated `delta`/`epsilon` length checks in `cavities.py` (covered by
  `validate_uq_config`, which `validate_tune_config` reaches through both the nested `uq_config` and
  `eigenmode_config`), and rewrote the entangled `processes`/`MT`/`NFS` blocks — which were
  `if key in cfg: assert ... else: cfg[key] = default` — as plain `setdefault`, since
  `validate_wakefield_config` already type-checks them. Guarded by
  `test_api_guards.py::test_no_asserts_in_the_package`.

- [x] **P3-4 (solver selection): capability-probed direct solver.** *(2026-07-09.)*
  `default_direct_solver()` now prefers `pardiso` on Windows and **`umfpack` on macOS/Linux**, but
  only after `direct_solver_available(name)` probes a 2x2 problem — which sparse direct solvers are
  compiled in varies by platform *and* by how NGSolve was built (pip wheel / conda / source), so
  guessing would crash users. Falls back to the always-built-in `sparsecholesky`. Cached
  (`lru_cache`), so the probe runs at most once per backend. Confirmed on this Windows/conda build:
  pardiso and sparsecholesky available, **umfpack and mumps are not** — which is exactly why the
  probe is needed rather than a bare platform check. Override with `eigenmode_config['direct_solver']`.

  *(History: the `require()` helper and the config validators landed 2026-07-04; the remaining 73
  asserts across `cavities.py`, `optimisation.py`, `dakota.py`, `pyTuner.py`, `base.py`, `rfgun.py`,
  `write_dakota_input.py`, `sampling.py` and the processes modules were converted 2026-07-09.)*

- [x] **P2-8: Multiprocessing on Windows.** *(Done 2026-07-05.)* Verified `processes=2` end to
  end on Windows (spawn): Cavity objects pickle and both cavities' results are written. Test
  `test_cavity_types.py::test_parallel_eigenmode_processes_gt_1`.

## P3 — Nice to have

- [x] **Geometry unification.** *(Complete 2026-07-09 — every model is native.)* Originally a
  prototype landed 2026-07-05; the full history is below. Remaining follow-up, tracked separately:
  have `Profile` emit the ABCI point list so one geometry source feeds eigenmode + wakefield, after
  which the ~3,150 duplicated contour-walking lines can collapse.

  New `cavsim2d/geometry/`:
  a backend-agnostic `Profile` blueprint (ordered meridian segments — lines/arcs — each tagged
  AXI/PEC/PMC) that meshes natively via `netgen.occ` (exact edges, no gmsh/.geo). The solver
  dispatches through `NGSolveMEVP._build_mesh(cav)`: native `Profile` path if the cavity exposes
  `profile()`, else the existing gmsh `.geo` import path (elliptical/spline/imported CAD) —
  unchanged. **Pillbox ported as the reference** (`Pillbox.profile()`): native eigenmode matches
  the gmsh path (1159.56 vs 1159.6 MHz); all pillbox flows (tuning/wakefield/multicell/parallel)
  pass. Key gotcha solved: OCC shape-healing strips edge names when a profile has collinear
  adjacent segments with different tags (a pillbox end plane: aperture PMC + plate PEC on the
  same z-plane), so boundaries are named on the *generated mesh* by point-on-segment matching,
  not on the OCC edges. Tests in `test_geometry.py`.

  **Elliptical ported 2026-07-09.** `Profile.ellipse_arc_to()` adds an *exact* conic segment
  (`Ellipse(gp_Ax2d(...), major, minor).Trim(t0, t1).Edge()`, short sweep between endpoints —
  no polyline approximation; validated to 8.8e-07 relative area error on a quarter-ellipse).
  `EllipticalCavity.profile()` covers the symmetric single-cell case (mid == end-left ==
  end-right, beampipe `both`/`none`), building iris arc → tangent line → equator arc and
  mirroring about the equator plane; it uses `tangent_coords` for the tangency points, so the
  alpha/tangency continuity the tuner relies on is carried over unchanged. Multicell,
  asymmetric end cells, one-sided beampipes and degenerate parameter sets return `None`, so
  `_build_mesh` transparently falls back to the gmsh importer. Native vs gmsh agreement:
  TESLA 1-cell 1300.1716 vs 1300.1714 MHz (0.000%), R/Q 113.466 vs 113.465 (0.001%); 800 MHz
  midcell 801.182 vs 801.192 MHz. `G` and `Epk/Eacc` differ by 0.08% / 0.5% — both are
  peak-field/surface samplers whose value depends on where mesh nodes fall, so a different
  mesh legitimately shifts them. Gotcha: `profile()` must read the live `self.parameters`
  dict (what `write_geometry` uses and what the tuner mutates in place), **not** the
  `self.mid_cell` array frozen at `__init__` — reading the array made the tuned frequency
  constant across candidates and broke root-finding.

  **Geometry consolidated into one package, 2026-07-09.** All geometry now lives under
  `cavsim2d/geometry/`, split by role: `profile.py` (the blueprint), `tangency.py` (alpha /
  tangent-point solvers), `primitives.py` (line + arc point sampling), `plotting.py`
  (matplotlib renderers), and `writers/` (`gmsh.py`, `abci.py`, `multipac.py`, `cst.py`).
  Deleted: `cavsim2d/utils/geometry.py` (a 4,302-line grab bag),
  `cavsim2d/analysis/wakefield/geometry.py`, `cavsim2d/analysis/wakefield/abci_geometry.py`.
  Non-geometry residue (shape-dict continuity, perturbation, tabulation) moved to
  `cavsim2d/utils/shapes.py`. `shared_functions.py` remains a re-export shim so existing user
  scripts keep working. Cavity classes stayed in `cavsim2d/cavity/` — the public API is
  unchanged.

  Import-cycle gotcha: `geometry/writers/abci.py` imports `abci_code`, which star-imports
  `shared_functions`, which imports back into `cavsim2d.geometry`. Both `geometry/__init__.py`
  and `geometry/writers/__init__.py` are therefore kept light and must **never** import `abci`.

  **`Pillbox.profile()` was missing** and has now been written (2026-07-09). The earlier
  "Pillbox ported as the reference" claim was wrong — no `profile()` existed in the working tree
  or at HEAD, so `test_pillbox_profile_native_matches_gmsh` had been passing via the *gmsh
  fallback* the whole time. Native now matches gmsh exactly (1-cell 1159.5603 MHz both ways;
  2-cell+beampipe 1153.1949 both ways — identical, since an all-line pillbox meshes the same).
  Tests now call `assert_meshes_natively()`, which points `geo_filepath` at a nonexistent file so
  a silent gmsh fallback cannot pass; verified it catches the original bug.

  **Models split out of `cavity/`, 2026-07-09.** `cavsim2d/models/` now holds *only* the
  simulatable structures — `base.py` (the `Cavity` ABC), `elliptical.py`,
  `elliptical_flattop.py`, `pillbox.py`, `rfgun.py`, `spline.py`, `circular_waveguide.py`.
  `cavsim2d/cavity/` keeps what is not a model: `cavities.py` (the container), `dakota.py`,
  `operating_points.py`, `welcome.py` — and re-exports every model, so
  `from cavsim2d.cavity import EllipticalCavity` is unchanged. Moved with `git mv`, so history
  follows the files. (Named `models`, plural, rather than `model`.)

  **All geometries rewritten on `Profile`, 2026-07-09.** Every model except `SplineCavity` now
  meshes natively; `_build_mesh` still falls back to gmsh for anything without a `profile()`.

  - `cavsim2d/geometry/contours.py` — new. `elliptical_profile()` builds any elliptical contour
    from one observation: **every cell is a forward half (iris arc -> tangent line -> equator arc)
    plus an optional flat top plus a backward half.** One builder therefore serves single-cell,
    multicell, asymmetric end cells, one-sided beampipes, *and* the flat-top parameterisation.
    Key enabler: `tangent_coords` offsets `x - L_bp` are translation-invariant (verified to 9
    decimals), so solving once at `L_bp = 0` gives offsets valid at any beampipe length.
  - `Profile.circle_arc_to(z, r, center)` — new exact circular-arc segment (places the arc
    midpoint on the circle, builds a 3-point `ArcOfCircle`). Used by the RF gun's 8 arcs.
  - `EllipticalCavity.profile()` — now covers multicell / asymmetric / one-sided beampipe.
    Returns `None` only when the tangent solve fails (`fsolve` non-convergence), so the gmsh
    writer reports degenerate sets. Note it does **not** validate physically-silly-but-solvable
    sets (e.g. `Req < Ri`).
  - `EllipticalCavityFlatTop.profile()`, `RFGun.profile()`, `CircularWaveguide.profile()`,
    `Pillbox.profile()` — all native.

  Validation, native vs gmsh (worst QOI deviation): elliptical 1-cell/2-cell/3-cell, asymmetric
  end cells, one-sided beampipe all <= 0.05%; frequency and R/Q agree to ~0.001%. Pillbox and
  circular waveguide agree *exactly*. Circular waveguide is within 0.036% of the analytic TM010.
  RF gun: freq 213.063 vs 213.063 MHz, G within 0.02%; geometry proven identical (area, perimeter
  and every tagged boundary length match to 1e-13).

  **Two bugs found while porting.**
  1. `EllipticalCavityFlatTop.write_geometry` **writes no file at all** — its `to_csv` is commented
     out — so `geo_filepath` pointed at a file that never existed and the flat-top could never run
     an eigenmode. `profile()` is the first thing that makes it work. Its only ground truth is the
     `l -> 0` limit, where it reproduces the plain elliptical cavity exactly (801.1818 MHz both
     ways, 0.0000%); frequency also falls monotonically with flat length, as it must. The dead
     `.geo` writer should be deleted or fixed.
  2. `mesh_h` default was **20 metres**, not 20 mm: `mesh_config['h']` was scaled by `1e-3` only
     when the user passed `h`. A 20 m maxh is no constraint at all, so each meshing backend fell
     back to its own element sizing — harmless while gmsh was the only backend, but it made
     default results backend-dependent as soon as the native path existed (it is what produced a
     48% spread in the gun's `Epk/Eacc`). Fixed at both sites in `eigen_ngsolve.py`; no reference
     value in the suite moved.

  Note `Epk/Eacc` for the RF gun does **not** converge under refinement on either backend
  (native 15.6 -> 15.8 -> 16.3 -> 22.0; gmsh 15.5 -> 15.4 -> 16.3 -> 12.9 as maxh goes 30 -> 10 mm)
  while `freq` and `G` converge cleanly. The gun contour has sharp corners where the surface field
  is singular, so that QOI is mesh noise. Do not trust it, and do not treat a native/gmsh
  disagreement in it as a porting error.

  **`SplineCavity` ported, 2026-07-09 — every model is now native.** `Profile.spline_to(poles,
  boundary, kind, degree)` adds a free-form segment with gmsh's control-point semantics
  (`'bezier'` or `'bspline'`; clamped, so it starts and ends on the contour but need not pass
  through the interior poles).

  Gotcha: **netgen's `BSplineCurve(points, degree)` is *unclamped*** — for the 6-pole test wall it
  starts at (0.005, 0.0697) rather than the first pole, so the wire will not close. Bezier arcs
  are clamped and correct. The fix is `_bspline_to_bezier()`: raise every internal knot to
  multiplicity `degree` by Boehm insertion, decomposing the clamped B-spline into exact Bezier
  arcs that netgen represents exactly (validated to 8e-17 against the clamped curve, and to 1e-17
  against gmsh's own sampled curve at both 1 and 2 cells). No approximation anywhere.

  Validation: Bezier walls match gmsh to 1e-4 % on freq/R/Q/G at 1 and 2 cells. B-spline 1-cell
  geometry matches gmsh to 4e-13 (area and wall length).

  **The `'Berzier'` default was silently producing a wall-less cavity.** `SplineCavity.__init__`
  defaulted `kind='Berzier'`, which matched neither the `'BSpline'` nor `'Bezier'` branch of
  `write_geometry`, so the `.geo` came out with no PEC boundary at all — a degenerate triangle
  (area 0.0020 vs 0.0089, boundaries `['AXI','PMC']`). Default fixed to `'Bezier'`, `'Berzier'`
  kept as an alias, and an unknown kind now raises instead of mis-meshing.

  Known limitation, documented not fixed: for a **multicell B-spline**, repeating the control
  polygon puts a genuine stationary corner at the iris (`|dC/du| = 0` exactly at the knot). The
  native path resolves it — its wall length matches the analytic arc length to 3e-7, whereas gmsh
  *rounds the corner off* and comes out 0.48% short, which is the whole of the 5.95% QOI gap
  between the two paths there. But netgen's high-order curving then fails (`Standard_OutOfRange`)
  at fine meshes (order 3, maxh <= 4 mm; order 4, maxh <= 6 mm). The default (maxh 20 mm, order 3)
  works. Prefer `kind='Bezier'` for multicell splines — it emits one curve per cell, has no
  stationary corner, and is unaffected at every mesh size tested.

  NEXT: have `Profile` emit the ABCI point list so one geometry source feeds both eigenmode and
  wakefield. Only after that can the ~3,150 lines of duplicated contour walking in
  `writers/gmsh.py` (1,719), `plotting.py` (653) and `writers/multipac.py` (783) be collapsed —
  they all trace the same contour and differ only in the sink they emit to. The ABCI writer is a
  full input deck with mesh densities, so only its geometry half collapses.

- [x] **P3-1: Mark experimental modules.** *(Done 2026-07-04.)* Added `.. note:: Experimental /
  unsupported` module docstrings to `quicktools.py`, `utils/sensitivity.py`, `utils/surrogate.py`
  and `cavity/dakota.py` so they don't set stable-API expectations.
- [~] **P3-2: Release machinery.** *(Partly done 2026-07-04.)* Added `CHANGELOG.md` (0.1.0
  Unreleased notes incl. a Known Issues section). STILL TODO: GitHub release + tag, PyPI
  publish (check the name is free), issue/PR templates, `CITATION.cff` (thesis-adjacent — make
  it citable).
- [~] **P3-3: API polish.** *(Partly done 2026-07-04.)* Added `__repr__` to `Cavity`
  (`EllipticalCavity(name='TESLA', n_cells=1, beampipe='none', results=eigenmode+tuned)`) and
  `Cavities` (`Cavities(project='...', 3 cavities: [...])`) — much friendlier in REPL/notebooks.
  STILL TODO: type hints on public entry points; consistent `run_*` return types (bool vs None);
  a docstring pass with one runnable example per public method.
- [~] **P3-4: Performance.** *(Partly done.)* `pardiso`/`sparsecholesky` selection via
  `direct_solver` (done earlier); PINVIT `maxit` now exposed as `eigenmode_config['pinvit_maxit']`
  (monopole). STILL TODO: factorisation reuse across tuner iterations, PINVIT `tol` knob,
  field-export downsampling for big meshes.
- [~] **P3-5: Jupyter niceties.** *(Partly done.)* Added `cavs.summary()` → a fundamental-mode
  QOI DataFrame that renders as an HTML table in Jupyter. STILL TODO: tqdm progress bars for
  parallel runs, `suppress_c_stdout_stderr` robustness when stdout isn't a real fd.

- **Bug fixes (2026-07-05):** (1) single-cavity `cav.run_eigenmode(uq_config=…)` read UQ from the
  wrong path (`eigenmode/uq.json`) — now reads `<cav>/uq/`; (2) the all-modes UQ column parser
  now matches the unified `name_<idx>-<m> [unit]` key format.

- **Tuner correctness (2026-07-24):** three real bugs fixed so a tuned cavity actually sits on
  its target. (1) `cav.tuned` reloaded the UNTUNED geometry (a 1300 MHz tune re-solved to
  1236.9): `_load_tuned_from_disk` merged the FILTERED per-stage snapshot from `tune_res.json`
  (mid-cell stage stores only `*_m`) onto untuned siblings, then `_unify_equator_radius` (which
  treats the end-cell as canonical for a single cell) wiped the tuned `Req_m` back to the start —
  now `_run_tune` persists the complete sibling-propagated `final_params` to
  `tune_info/tuned_parameters.json` and the reload PREFERS it. (2) A mid-cell (periodic) tune
  leaked the cavity's beampipe — the eigensolver meshes `cav.profile()` (reads `cav.beampipe`),
  not the reduced `.geo` — so `beampipe='both'` gave the wrong `Req`; `tune_function` now pins
  `cav.beampipe` to the stage's beampipe around the solve. (3) End cells are tuned via the
  half-length `L`, not `Req` (which is unified across cells). Guards:
  `test_tune.py::{test_tuned_cavity_actually_sits_on_target_frequency,test_mid_cell_tune_is_beampipe_independent}`.
  New examples: `tuning/{multicell,cavity_types}.ipynb` (mid+end multicell tuning to a flat-field
  pi-mode; one `run_tune` call across four cavity types). See memory `tuning-mechanics`.

---

## Suggested order of attack

1. P0-1 (physics bug) → immediately re-baseline QOIs.
2. P0-2 (commit everything) — do this today.
3. P0-3 (delete broken legacy layer) — shrinks the audit surface for everything below.
4. P1-5 (portable tests + CI green) — locks in the verified flows before further changes.
5. P1-1/P0-5 (packaging + licence) → P1-2 (README) → P1-6 (notebooks) → P1-3/P1-4 (config UX).
6. P1-7, P1-8 → tag v1.0.0-rc1, ask 1–2 friendly colleagues to install from scratch and follow
   the README quickstart on their machines (the single highest-value pre-ship test).
7. P2 items as post-rc follow-ups; P3 opportunistically.
