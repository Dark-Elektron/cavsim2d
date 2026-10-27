"""Run independent per-cavity tasks on worker processes, with a progress bar."""
from tqdm.auto import tqdm


def run_each(function, tasks, processes, desc):
    """Call ``function(*task)`` for every argument tuple in *tasks* on *processes*
    worker processes and return the results in task order.

    Workers come from joblib's loky backend, which, unlike
    ``multiprocessing.Process`` with Windows' spawn start, runs inside a Jupyter
    notebook or a script without an ``if __name__ == '__main__'`` guard. One task
    per cavity lets the workers share the load unevenly sized cavities make, and
    the bar advances as each one finishes.
    """
    # Deferred: joblib is only pulled in for parallel runs.
    from joblib import Parallel, delayed
    tasks = list(tasks)
    runs = Parallel(n_jobs=processes, backend='loky', return_as='generator')(
        delayed(function)(*task) for task in tasks)
    return list(tqdm(runs, total=len(tasks), desc=desc, unit='cavity'))
