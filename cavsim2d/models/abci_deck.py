"""The ABCI wakefield deck of a cavity, and the mesh steps it is written with.

:class:`~cavsim2d.models.base.Cavity` inherits these from :class:`AbciDeck`.
"""
import re
from cavsim2d.constants import MROT_DICT
from cavsim2d.utils.printing import info, warning
from cavsim2d.utils.shapes import make_dirs_from_dict
import ast
import operator as op
import os
from cavsim2d.geometry.beampipes import abci_shape
from cavsim2d.solvers.ABCI.abci import resolve_mrot


# Safe arithmetic evaluator for simple expressions
_ops = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul,
        ast.Div: op.truediv, ast.Pow: op.pow, ast.USub: op.neg}


#: ABCI mesh step [m] used when neither an explicit DDR/DDZ nor a short bunch asks
#: for a finer one.
ABCI_DEFAULT_STEP = 0.00125


def abci_bunch_length(wakefield_config):
    """RMS bunch length [m] a wakefield config asks for, nested (``beam_config``)
    or top-level; ``None`` when it sets none."""
    cfg = wakefield_config or {}
    bunch = (cfg.get('beam_config') or {}).get('bunch_length', cfg.get('bunch_length'))
    return None if bunch is None else float(bunch) * 1e-3


def abci_mesh_steps(wakefield_config):
    """ABCI's radial and axial mesh steps ``(DDR, DDZ)`` in metres.

    An explicit ``mesh_config['DDR']``/``['DDZ']`` wins. Otherwise the step is
    ``DDR_SIG``/``DDZ_SIG`` (default 0.1) times the bunch length, capped at
    :data:`ABCI_DEFAULT_STEP`. Those two ratios were documented as the mesh control
    but never read, so the mesh stayed at 1.25 mm whatever the bunch. That is 20
    steps per sigma for the default 25 mm bunch, but only 3.5 for a 4.32 mm one,
    too coarse to resolve it. The cap keeps every run with a bunch of 12.5 mm or
    longer on exactly the mesh it always had.
    """
    cfg = wakefield_config or {}
    mesh_cfg = cfg.get('mesh_config') or {}
    sig = abci_bunch_length(cfg)
    steps = []
    for key in ('DDR', 'DDZ'):
        if mesh_cfg.get(key) is not None:
            steps.append(float(mesh_cfg[key]))
            continue
        ratio = mesh_cfg.get(f'{key}_SIG', cfg.get(f'{key}_SIG', 0.1))
        step = ABCI_DEFAULT_STEP
        if sig is not None and ratio:
            step = min(step, float(ratio) * sig)
        steps.append(step)
    return tuple(steps)


class AbciDeck:
    """Writing a cavity's ABCI wakefield deck (from its profile or its ``.geo``)."""

    def eval_expr(self, expr, symbols):
        """Evaluate numeric expression with symbols."""
        node = ast.parse(expr, mode='eval').body

        def _eval(n):
            # ast.Constant, not ast.Num: Num and its .n are gone in Python 3.14.
            if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
                return n.value
            if isinstance(n, ast.Name): return symbols[n.id]
            if isinstance(n, ast.BinOp): return _ops[type(n.op)](_eval(n.left), _eval(n.right))
            if isinstance(n, ast.UnaryOp): return _ops[type(n.op)](_eval(n.operand))
            raise ValueError(f"Unsupported expression: {expr}")

        return _eval(node)

    # Main converter
    def geo_to_abc(self, wakefield_config=None, folder=None, **kwargs):
        """Write this cavity's ABCI input deck(s).

        The contour comes from the unified :class:`~cavsim2d.geometry.Profile`
        (densified to the mesh spacing, exact for ellipse / circle / spline
        walls), with beam pipes guaranteed at both ends because ABCI refuses to
        run without them. Cavities that expose no ``profile()`` — an imported
        ``.geo``/CAD — fall back to parsing the ``.geo`` text.
        """
        self.create()

        cfg = wakefield_config or {}
        ddr, ddz = abci_mesh_steps(cfg)
        ds = float(cfg.get('contour_ds', min(ddr, ddz)))
        # ABCI wants at least 5 mesh lengths of pipe; ask for a little margin.
        min_pipe = 6.0 * ddz
        pipe_length = cfg.get('beampipe_length')      # metres; None -> 3x axial length

        maker = getattr(self, 'profile', None)
        profile = maker() if callable(maker) else None

        if profile is not None:
            # ABCI's own arc primitive is kept: replacing an arc that meets a beam
            # pipe tangentially (an elliptical cavity's iris) with a dense polyline
            # makes ABCI's mesher emit NaN wake potentials.
            items, added_l, added_r = abci_shape(profile, ds, min_pipe, pipe_length)
            if added_l or added_r:
                info(f"{self.name}: ABCI needs a beam pipe at each end; added "
                     f"{added_l * 1e3:.1f} mm on the left and {added_r * 1e3:.1f} mm on "
                     f"the right.")
            zmin = min(it[1] if it[0] == 'point' else it[2][0] for it in items)
            segments = []
            for it in items:
                if it[0] == 'point':
                    segments.append(('point', it[2], it[1] - zmin))
                else:
                    (zc, rc), (ze, re) = it[1], it[2]
                    segments.append(('arc', rc, zc - zmin, re, ze - zmin))
            points = {}
        else:
            points, segments, zmin = self._contour_from_geo()

        wakefield_folder_structure = {
            'wakefield': {
                'longitudinal': None,
                'transversal': None
            }
        }
        if folder:
            make_dirs_from_dict(wakefield_folder_structure, folder)
        else:
            make_dirs_from_dict(wakefield_folder_structure, self.self_dir)
            folder = self.self_dir

        MROT = resolve_mrot(wakefield_config)
        if MROT == 2:
            for m in range(2):
                self._write_abc(points, segments, zmin, {}, m, wakefield_config, folder, **kwargs)
        else:
            self._write_abc(points, segments, zmin, {}, MROT, wakefield_config, folder, **kwargs)

    def _contour_from_geo(self):
        """Legacy contour extraction: regex-parse the gmsh ``.geo`` text.

        Only understands ``Point(...)`` and ``Ellipse(...)``, and encodes each
        ellipse as an ABCI ``-3.`` *circular* arc about the ellipse centre — which
        is only faithful when the ellipse is a circle. Kept for cavities with an
        imported ``.geo`` and no ``profile()``.
        """
        with open(self.geo_filepath) as f:
            lines = [l.split('%', 1)[0].strip() for l in f]

        symbols = {}
        define_re = re.compile(r"(\w+)\s*=\s*DefineNumber\[\s*([^,\]]+)")
        for line in lines:
            m = define_re.search(line)
            if m:
                symbols[m.group(1)] = m.group(2).strip()
        changed = True
        while changed:
            changed = False
            for k, v in list(symbols.items()):
                if isinstance(v, str):
                    try:
                        symbols[k] = self.eval_expr(v, symbols)
                        changed = True
                    except Exception:
                        pass

        point_re = re.compile(r"Point\((\d+)\)\s*=\s*\{([^}]+)\}")
        ellipse_re = re.compile(r"Ellipse\(\d+\)\s*=\s*\{([^}]+)\}")

        points = {}
        segments = []
        zvals = []
        for line in lines:
            m = point_re.match(line)
            if m:
                pid = int(m.group(1))
                coords = [p.strip() for p in m.group(2).split(',')]
                z = self.eval_expr(coords[0], symbols)
                r = self.eval_expr(coords[1], symbols)
                points[pid] = (r, z)
                zvals.append(z)

        zmin = min(zvals) if zvals else 0.0

        for line in lines:
            if line.startswith('Point('):
                m = point_re.match(line)
                pid = int(m.group(1))
                r, z = points[pid]
                segments.append(('point', r, z - zmin))
            elif line.startswith('Line('):
                continue
            elif line.startswith('Ellipse('):
                m = ellipse_re.match(line)
                ids = [int(x.strip()) for x in m.group(1).split(',')]
                if len(segments) >= 3:
                    segments = segments[:-3]
                segments.append(('ellipse', ids[1], ids[3]))

        return points, segments, zmin

    def _write_abc(self, points, segments, zmin, symbols, MROT, wakefield_config, folder, **kwargs):
        # defaults
        RDRIVE, ISIG = 5e-3, 5
        LCRBW = 'F'  # counter-rotating beam
        ZSEP = 0.0
        BSEP = 0
        NBUNCH = 1
        BETA = 1
        LMATPR = '.F.'
        LPRW, LPPW, LSVW, LSVWA, LSVWT, LSVWL, LSVF = '.T.', '.T.', '.T.', '.F.', '.T.', '.T.', '.F.'
        LSAV, LCPUTM = 'F', 'F'
        LCBACK = '.T.'
        LPLE = '.F.'
        NSHOT = 0
        UBT = 50
        SIG = 25e-3
        MT = 4

        # unpack kwargs
        for key, value in kwargs.items():
            if key == 'RADIAL BEAM OFFSET AT (RDRIVE)':
                RDRIVE = value
            if key == 'NUMBER OF WAKE POTENTIAL POINTS (NW)':
                NW = value
            if key == 'WAKE FOR A COUNTER-ROTATING BEAM (LCRBW)':
                LCRBW = value
            if key == 'VELOCITY OF THE BUNCH / C (BETA)':
                BETA = value
            if key == 'PRINTOUT OF CAVITY SHAPE USED (LMATPR)':
                LMATPR = value
            if key == 'PRINTOUT OF WAKE POTENTIALS (LPRW)':
                LPRW = value
            if key == 'LINE-PRINTER PLOT OF WAKE POT. (LPPW)':
                LPPW = value
            if key == 'SAVE WAKE POTENTIALS IN A FILE (LSVW)':
                LSVW = value
            if key == 'SAVE AZIMUTHAL WAKE IN A FILE (LSVWA)':
                LSVWA = value
            if key == 'SAVE TRANSVERSE WAKE IN A FILE (LSVWT)':
                LSVWT = value
            if key == 'SAVE LONGITUDINAL WAKE IN A FILE (LSVWL)':
                LSVWL = value
            if key == 'SAVE FFT RESULTS IN A FILE (LSVF)':
                LSVF = value
            if key == 'SAVE FIELDS INTO FILE (LSAV)':
                LSAV = value
            if key == 'CPUTIME MONITOR ACTIVE (LCPUTM)':
                LCPUTM = value

        # ABCI's own field mesh. geo_to_abc already reads these to size the
        # contour sampling and the minimum beam pipe, so leaving them out here
        # left the deck pinned at the default: a caller who set DDR/DDZ got a
        # differently *sampled* wall solved on an unchanged mesh, and a mesh
        # convergence check came back perfectly flat. The same helper sizes both,
        # so the wall sampling and the field mesh cannot disagree.
        mesh_DDR, mesh_DDZ = abci_mesh_steps(wakefield_config)

        if 'save_fields' in wakefield_config.keys():
            LPLE, LCBACK = 'T', 'F'
            if isinstance(wakefield_config['save_fields'], dict):
                if 'nshot' in wakefield_config['save_fields'].keys():
                    NSHOT = wakefield_config['save_fields']['nshot']

        if 'wake_config' in wakefield_config.keys():
            wake_config = wakefield_config['wake_config']
            if 'counter_rotating'in wakefield_config['wake_config'].keys():
                LCRBW = 'T'
                if 'separation' in wakefield_config['wake_config']['counter_rotating'].keys():
                    ZSEP = wakefield_config['wake_config']['counter_rotating']['separation']

        if 'beam_config' in wakefield_config.keys():
            if 'beam_offset' in wakefield_config['beam_config'].keys():
                RDRIVE = wakefield_config['beam_config']['beam_offset']
            if 'nbunch' in wakefield_config['beam_config'].keys():
                NBUNCH = wakefield_config['beam_config']['nbunch']
            if 'separation' in wakefield_config['beam_config'].keys():
                BSEP = wakefield_config['beam_config']['separation']

        # Bunch length (SIG) and wake length (UBT). Both are accepted either
        # nested under beam_config/wake_config — the form Study.run_wakefield
        # populates — or at the top level, the form DEFAULT_WAKEFIELD_CONFIG
        # declares. Without this the hardcoded defaults above were the only
        # values that ever reached the deck, so every loss factor came back at
        # 25 mm whatever the caller asked for.
        bunch_length = (wakefield_config.get('beam_config') or {}).get(
            'bunch_length', wakefield_config.get('bunch_length'))
        if bunch_length is not None:
            SIG = float(bunch_length) * 1e-3          # [mm] -> [m]

        wakelength = (wakefield_config.get('wake_config') or {}).get(
            'wakelength', wakefield_config.get('wakelength'))
        if wakelength is not None:
            UBT = float(wakelength)                   # [m]

        # MT (time steps to cross a mesh cell -> time-integration resolution). Accept it
        # nested OR top-level, like SIG/UBT above. It used to be read only from
        # wake_config['MT'], so a top-level MT -- the form run_wakefield and
        # DEFAULT_WAKEFIELD_CONFIG use -- was silently dropped and the run fell back to the
        # MT=4 default. That coarse integration degrades the TRANSVERSE (dipole) wake far
        # more than the on-axis longitudinal monopole (noisy/smeared transverse spectrum).
        mt_cfg = (wakefield_config.get('wake_config') or {}).get('MT', wakefield_config.get('MT'))
        if mt_cfg is not None:
            MT = int(mt_cfg)
        # ABCI caps its time mesh at NPT=40, i.e. 2*(MT+1) <= 40 -> MT <= 19. A larger MT
        # aborts the run ("2*(MT+1) EXCEEDS MAXIMUM NPT"), so clamp instead. (Empirically
        # MT barely moves the impedance spectrum anyway -- MT=4 and MT=15 are identical.)
        if MT > 19:
            warning(f'MT={MT} exceeds the ABCI time-mesh limit (2*(MT+1) <= NPT=40); '
                    f'clamping to MT=19.')
            MT = 19

        # Upper frequency of the FFT/spectrum output [GHz]. Without it ABCI
        # runs the spectrum out to its own limit — tens of GHz for a short
        # bunch — which is mostly noise and makes the output files large.
        CUTOFF = (wakefield_config.get('wake_config') or {}).get(
            'cutoff', wakefield_config.get('cutoff'))

        with open(os.path.join(folder, 'wakefield', MROT_DICT[MROT], 'cavity.abc'), 'w') as out:
            # Header
            out.write(f' &FILE LSAV = .{LSAV}., ITEST = 0, LREC = .F., LCPUTM = .{LCPUTM}. &END \n')
            out.write(' SAMPLE INPUT #1 A SIMPLE CAVITY STRUCTURE \n')
            out.write(' &BOUN  IZL = 3, IZR = 3  &END \n')
            out.write(f' &MESH DDR = {mesh_DDR}, DDZ = {mesh_DDZ} &END \n')
            out.write(" #CAVITYSHAPE\n0.\n")

            # Body
            for seg in segments:
                if seg[0] == 'point':
                    _, r, z = seg
                    out.write(f"{r:.16f} {z:.16f}\n")
                elif seg[0] == 'arc':
                    # '-3.' introduces an arc given by its centre, then its end point.
                    # Emitted by the Profile path; already z-shifted.
                    _, r_c, z_c, r_e, z_e = seg
                    out.write("-3., 0.000\n")
                    out.write(f"{r_c:.16f} {z_c:.16f}\n")
                    out.write(f"{r_e:.16f} {z_e:.16f}\n")
                elif seg[0] == 'ellipse':
                    # legacy .geo path: point ids into `points`
                    _, pid_m, pid_e = seg
                    r_m, z_m = points[pid_m]
                    r_e, z_e = points[pid_e]
                    out.write("-3., 0.000\n")
                    out.write(f"{r_m:.16f} {z_m - zmin:.16f}\n")
                    out.write(f"{r_e:.16f} {z_e - zmin:.16f}\n")

            # Closing
            out.write("0.000 0.000\n9999. 9999.\n")

            # wakefield simulation paramters
            out.write(f' &BEAM  SIG = {SIG}, ISIG = {ISIG}, RDRIVE = {RDRIVE}, MROT = {MROT}, NBUNCH = {NBUNCH}, BSEP = {BSEP} &END \n')
            # out.write(' &BEAM  SIG = {}, MROT = {}, RDRIVE = {}  &END \n'.format(SIG, MROT, 0.005))
            out.write(f' &TIME  MT = {int(MT)} &END \n')
            # UBT must not be truncated to an integer: a loss-factor run uses a
            # wake only a few bunch lengths long (5 x 2.4 mm = 0.012 m), and
            # int() turned that into 0.
            out.write(
                f' &WAKE  UBT = {UBT:g}, LCRBW = .{LCRBW}., LCBACK = {LCBACK}, LCRBW = .{LCRBW}., ZSEP = {ZSEP} &END \n')
            # No NFS here. ABCI has no such namelist variable -- its &WAKE takes
            # UBT, LCFRON, LCBACK, LCHIN, LNAPOLY, ZCF, ZCT, RWAK, LCRBW, ZSEP,
            # LNONAP and nothing else, and an unknown name is a namelist read
            # error. The number of points in the FFT output is ABCI's own, and
            # its resolution is set by the wake length (df ~ c/UBT); CUTOFF in
            # &PLOT below bounds the range. A dead 'NFS = {NFS}' fragment used to
            # sit at the end of this line, which made the config key look live.
            # f.write(' &WAKE  UBT = {}, LCHIN = F, LNAPOLY = F, LNONAP = F &END \n'.format(UBT, wake_offset))
            # f.write(' &WAKE R  = {}   &END \n'.format(wake_offset))
            _cutoff = f'CUTOFF = {CUTOFF:g}, ' if CUTOFF is not None else ''
            out.write(f' &PLOT  LCAVIN = .T., LCAVUS = .F., LPLW = .T., LFFT = .T., LSPEC = .T., '
                    f'{_cutoff}LINTZ = .F., LPATH = .T., LPLE = {LPLE}, LPLC= .F. &END \n')
            out.write(f' &PRIN  LMATPR = {LMATPR}, LPRW = {LPRW}, LPPW = {LPPW}, LSVW = {LSVW}, '
                    f'LSVWA = {LSVWA}, LSVWT = {LSVWT}, LSVWL = {LSVWL},  LSVF = {LSVF}   &END\n')
            out.write('\nSTOP\n')
