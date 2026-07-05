"""Run every cavsim2d example script in order.

Usage:
    python run_all.py            # run all examples
    python run_all.py 1 2 5      # run only examples 1, 2 and 5

Each example writes its results and plots under
    <SIM_ROOT>/examples/<name>/
(see _common.SIM_ROOT). The wakefield example self-skips off Windows.
"""
import importlib
import sys
import time

from _common import banner

EXAMPLES = [
    ('01_eigenmode', 'Monopole eigenmode'),
    ('02_eigenmode_mpole', 'm-pole eigenmodes'),
    ('03_tune', 'Frequency tuning'),
    ('04_wakefield', 'Wakefield / impedance'),
    ('05_eigenmode_uq', 'Eigenmode + UQ'),
    ('06_optimisation', 'Multi-objective optimisation'),
]


def main(argv):
    selected = set(argv)
    ran = []
    for i, (module_name, title) in enumerate(EXAMPLES, start=1):
        if selected and str(i) not in selected:
            continue
        t0 = time.time()
        try:
            mod = importlib.import_module(module_name)
            mod.main()
            ran.append((title, time.time() - t0, 'ok'))
        except Exception as e:  # noqa: BLE001 - keep going through the suite
            import traceback
            traceback.print_exc()
            ran.append((title, time.time() - t0, f'FAILED: {e!r}'))

    banner("Summary")
    for title, dt, status in ran:
        print(f"  {title:<32} {dt:6.1f}s   {status}")


if __name__ == '__main__':
    main(sys.argv[1:])
