import os
import subprocess

from cavsim2d.constants import SOFTWARE_DIRECTORY, MROT_DICT


class ABCI:
    def solve(self, cav, wakefield_config, subdir=''):
        MROT = wakefield_config['polarisation']

        LCPUTM = 'F'
        if 'LCPUTM' in wakefield_config.keys():
            LCPUTM = wakefield_config['LCPUTM']

        exe_path = os.path.join(SOFTWARE_DIRECTORY, 'solvers', 'ABCI', 'ABCI.exe')

        if MROT == 2:
            for m in range(2):
                self.run_abci(cav, exe_path, LCPUTM, m, subdir)
        else:
            self.run_abci(cav, exe_path, LCPUTM, MROT, subdir)

    def run_abci(self, cav, exe_path, LCPUTM, MROT, subdir):
        if LCPUTM == 'T':
            subprocess.call([exe_path, os.path.join(cav.wakefield_dir, subdir, MROT_DICT[MROT], 'cavity.abc')])
        else:
            print(os.path.join(cav.wakefield_dir, subdir, MROT_DICT[MROT], 'cavity.abc'))
            subprocess.call([exe_path, os.path.join(cav.wakefield_dir, subdir, MROT_DICT[MROT], 'cavity.abc')],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)