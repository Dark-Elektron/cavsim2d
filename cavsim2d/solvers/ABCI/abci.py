import os
import platform
import shutil
import subprocess

from cavsim2d.constants import SOFTWARE_DIRECTORY, MROT_DICT


def abci_exe_path():
    """Absolute path to the bundled ABCI executable."""
    return os.path.join(SOFTWARE_DIRECTORY, 'solvers', 'ABCI', 'ABCI.exe')


_MROT_NAMES = {'monopole': 0, 'longitudinal': 0,
               'dipole': 1, 'transverse': 1, 'transversal': 1,
               'both': 2, 'all': 2}


def resolve_mrot(wakefield_config, default=2):
    """Resolve the ABCI beam rotation mode (MROT) from a wakefield config.

    ``MROT`` is the canonical key; ``polarisation`` is accepted as a
    deprecated alias — note it means the ABCI beam mode here (0 longitudinal /
    monopole, 1 transverse / dipole, 2 both), NOT the eigenmode azimuthal
    order. Either key may be given as one of those integers or the equivalent
    name.
    """
    if 'MROT' in wakefield_config:
        value = wakefield_config['MROT']
    elif 'polarisation' in wakefield_config:
        value = wakefield_config['polarisation']
    else:
        return default

    if isinstance(value, str):
        key = value.strip().lower()
        if key in _MROT_NAMES:
            return _MROT_NAMES[key]
        raise ValueError(f"Unknown wakefield MROT '{value}'. Use 0 (longitudinal), "
                         f"1 (transverse), 2 (both), or one of {sorted(_MROT_NAMES)}.")
    if value in (0, 1, 2):
        return int(value)
    raise ValueError("Wakefield MROT must be 0 (longitudinal), 1 (transverse) or "
                     "2 (both), or the equivalent name.")


def run_abci_exe(exe_path, input_path, run_dir, quiet=False):
    """Launch the ABCI solver on *input_path* with *run_dir* as cwd.

    ABCI.exe is a Windows binary. On Windows it is launched directly; on other
    platforms it is run through ``wine`` when available. A missing executable
    or missing ``wine`` raises a clear error instead of a cryptic OSError.

    NB: ABCI writes ``cavity.top`` (and auxiliary files) to its current working
    directory — not to the input file's directory — so *run_dir* must be where
    the downstream ABCIData reader looks for the output.
    """
    if not os.path.exists(exe_path):
        raise FileNotFoundError(
            f"ABCI solver not found at {exe_path}. Wakefield analysis needs the free "
            f"ABCI executable — see the README 'Third party code' section for how to "
            f"download and place it as cavsim2d/solvers/ABCI/ABCI.exe.")

    if platform.system() == 'Windows':
        cmd = [exe_path, str(input_path)]
    else:
        wine = shutil.which('wine')
        if wine is None:
            raise EnvironmentError(
                f"Wakefield analysis uses the Windows ABCI executable, which needs "
                f"'wine' on {platform.system()}. Install wine (https://www.winehq.org) "
                f"or run wakefield analysis on Windows.")
        cmd = [wine, exe_path, str(input_path)]

    kwargs = {'cwd': str(run_dir)}
    if quiet:
        kwargs.update(stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.call(cmd, **kwargs)


class ABCI:
    def solve(self, cav, wakefield_config, subdir=''):
        MROT = resolve_mrot(wakefield_config)

        LCPUTM = 'F'
        if 'LCPUTM' in wakefield_config.keys():
            LCPUTM = wakefield_config['LCPUTM']

        exe_path = abci_exe_path()

        if MROT == 2:
            for m in range(2):
                self.run_abci(cav, exe_path, LCPUTM, m, subdir)
        else:
            self.run_abci(cav, exe_path, LCPUTM, MROT, subdir)

    def run_abci(self, cav, exe_path, LCPUTM, MROT, subdir):
        input_path = os.path.join(cav.wakefield_dir, subdir, MROT_DICT[MROT], 'cavity.abc')
        run_dir = os.path.dirname(input_path)
        run_abci_exe(exe_path, input_path, run_dir, quiet=(LCPUTM != 'T'))