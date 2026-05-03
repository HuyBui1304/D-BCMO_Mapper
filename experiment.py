"""Notebook helpers for D_BCMO-Mapper experiments.

- ``auto_seed``: read the random seed from a ``.seed`` file in the working directory.
- ``dmapper_objective``: decorator that wraps boilerplate (seeding, param validation,
  error handling) around a notebook's scoring function.
- ``run_bcmo``: run BCMO once and log timing and best parameters.
"""
import os
import time

import numpy as np

from .bcmo import BCMO


__all__ = ["auto_seed", "dmapper_objective", "run_bcmo"]


def _read_seed_file(phase):
    cfg_path = os.path.join(os.getcwd(), ".seed")
    with open(cfg_path) as f:
        content = f.read().strip()
    if "=" in content:
        cfg = dict(
            line.strip().split("=", 1)
            for line in content.split("\n")
            if "=" in line
        )
        return int(cfg[phase])
    return int(content)


def auto_seed(phase="search"):
    """Seed ``np.random`` from ``./.seed``.

    Only ``numpy.random`` is seeded. Python ``random`` (used by ``randrange``
    and ``uniform`` inside the BCMO core) is intentionally left unseeded so
    each run has slightly different BCMO trajectories even with a fixed numpy
    seed, which is consistent with the original metaheuristic design.

    ``.seed`` file formats:

    - Single integer line → used for every phase.
    - Multiple ``key=value`` lines → looked up by ``phase``.
    """
    np.random.seed(_read_seed_file(phase))


def dmapper_objective(build_score):
    """Decorator wrapping a notebook scoring function ``(n, a) -> SC_adj``.

    Adds: seeding, parameter parsing/rounding, bounds validation, and
    error handling. Returns sentinel values on failure:
    - ``5.0`` — parameter coercion error
    - ``2.0`` — parameter out of valid range
    - ``3.0`` — Mapper / evaluation crash
    """
    def wrapped(X):
        auto_seed()
        try:
            n = int(round(float(X[0])))
            a = round(float(X[1]), 4)
        except Exception as e:
            print(f"[ERROR 1] Parameter coercion failed: params={X}, error={e}")
            return 5.0
        if n < 2 or n > 30 or a <= 0 or a >= 1:
            print(f"[ERROR 2] Parameter out of range: n={n}, a={a}")
            return 2.0
        try:
            return -build_score(n, a)
        except Exception as e:
            print(f"[ERROR 3] Mapper/SC_adj failed: n={n}, a={a}, error={e}")
            return 3.0
    return wrapped


def run_bcmo(objective_fn, bounds, NP=20, gen=10, run_id=1, output=None):
    """Run BCMO once and log timing and best parameters to ``output``.

    The numpy random state is NOT reset on entry, so back-to-back calls
    produce different BCMO trajectories even with the same ``.seed`` file.
    This is intentional: it trades strict reproducibility for broader
    landscape exploration across repeated runs.

    Parameters
    ----------
    objective_fn : callable
        Wrapped objective (output of ``dmapper_objective``).
    bounds : list of (low, high)
        Search bounds for ``[n_cubes, alpha]``.
    NP : int
        BCMO population size.
    gen : int
        Number of BCMO generations.
    run_id : int
        Run index appended to the log file.
    output : str or None
        Path to the append-only results file (``<dataset>_BCMO.txt``).
        If provided, a matching ``<dataset>_history.csv`` is also written.

    Returns
    -------
    best : ndarray
        Best individual found (shape ``(D+1,)``; last element is fitness).
    sc_adj : float
        Best ``SC_adj`` value (positive).
    """
    start_total = time.time()
    start_bcmo = time.time()
    X, history = BCMO(
        min_func=objective_fn, bounds=bounds, NP=NP, gen=gen, verbose=False
    )
    bcmo_duration = time.time() - start_bcmo

    best = X[0]
    n_best = int(round(best[0]))
    a_best = round(float(best[1]), 4)
    sc_adj = -objective_fn(best)

    total_duration = time.time() - start_total

    best_value = history[0][0]
    last_update_gen = 0
    for i in range(1, len(history)):
        if history[i][0] < best_value - 1e-12:
            best_value = history[i][0]
            last_update_gen = i
    gen_duration = bcmo_duration / len(history)
    convergence_time = gen_duration * last_update_gen

    if output:
        with open(output, "a") as f:
            f.write(f"Run {run_id}:\n")
            f.write(f"  Best n = {n_best}, a = {a_best}\n")
            f.write(f"  SC_adj = {sc_adj:.6f}\n")
            f.write(f"  BCMO time         = {bcmo_duration:.2f} seconds\n")
            f.write(f"  Total runtime     = {total_duration:.2f} seconds\n")
            f.write(f"  Convergence time  = {convergence_time:.2f} seconds (at generation {last_update_gen})\n")
            f.write("-" * 50 + "\n")

        hist_path = output.replace("_BCMO.txt", "_history.csv")
        new_file = not os.path.exists(hist_path)
        with open(hist_path, "a") as f:
            if new_file:
                f.write("run_id,gen,best_sc_adj\n")
            running_best = float("-inf")
            for g, (b, _w) in enumerate(history):
                sc_at_g = -b
                if sc_at_g > running_best:
                    running_best = sc_at_g
                f.write(f"{run_id},{g},{running_best:.6f}\n")
    return best, sc_adj
