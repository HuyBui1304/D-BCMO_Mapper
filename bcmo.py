"""
Balancing Composite Motion Optimization (BCMO).

Reference:
    Le-Duc, T., Nguyen, Q.-H., & Nguyen-Xuan, H. (2020).
    Balancing composite motion optimization.
    Information Sciences, 520, 250-270.

Equations below refer to the Le-Duc 2020 paper.
"""
from random import randrange, uniform

import numpy as np


__all__ = ["BCMO", "initial", "algorithm_1", "algorithm_2"]


def _project_int(x, int_dims):
    """Round the dimensions listed in ``int_dims`` to integers, in-place."""
    if int_dims:
        for d in int_dims:
            x[d] = round(x[d])
    return x


def initial(NP, bounds, min_func, int_dims=None):
    """Uniform random initialization of NP individuals (Eq. 1) and
    ranking by fitness (Eq. 2).

    Dimensions in ``int_dims`` are rounded to integers (paper Algo 3.1 line 12).
    """
    D = len(bounds)
    X = []
    while len(X) < NP:
        x = []
        for i in range(D):
            xi_candidate = bounds[i][0] + float(np.random.rand(1, 1)) * (bounds[i][1] - bounds[i][0])
            x.append(xi_candidate)
        x_arr = np.array(x).reshape(1, D)
        _project_int(x_arr[0], int_dims)
        fx = min_func(x_arr[0])
        x = list(x_arr[0]) + [fx]
        X.append(x)
    X = np.array(X)
    X = X[X[:, -1].argsort()]
    return X


def algorithm_1(Xt_1, bounds, min_func, int_dims=None):
    """Determine the instant global point O_in and the best individual
    x_1^t at generation t (Le-Duc Algorithm 1, Eqs. 4-6).
    """
    D = len(bounds)
    NP = Xt_1.shape[0]
    uc = [(i[0] + i[1]) / 2 for i in bounds]

    k1 = randrange(1, NP)
    k2 = randrange(0, k1)

    dvj = np.random.rand(1, D)[0]
    dvj = dvj if np.random.rand() > 0.5 else -dvj

    alphaij = dvj
    vk1_k2 = alphaij * (Xt_1[k2, :-1] - Xt_1[k1, :-1])
    vk2_1 = alphaij * (Xt_1[k2, :-1] - Xt_1[0, :-1])
    u1_candidate = uc + vk1_k2 + vk2_1

    u1 = np.clip(u1_candidate, [b[0] for b in bounds], [b[1] for b in bounds])
    _project_int(u1, int_dims)
    min1 = min_func(u1)
    min2 = min_func(Xt_1[0, :-1])

    if min1 <= min2:
        out = np.append(u1, min1)
    else:
        out = np.append(Xt_1[0, :-1], min2)

    Xt = Xt_1.copy()
    Xt[0, :] = out
    return Xt


def algorithm_2(Xt, bounds, min_func, int_dims=None):
    """Composite motion update for individuals i = 2..NP
    (Le-Duc Algorithm 2, Eqs. 7-15).
    """
    D = len(bounds)
    NP = Xt.shape[0]
    d = D

    Xt1 = [Xt[0, :]]

    for i in range(1, NP):
        j = randrange(0, i)
        TVj = uniform(0, 1)
        rj = abs(Xt[0, :-1] - Xt[j, :-1])

        Lgs = np.exp(-j / d * ((j / NP) if TVj > 0.5 else (1 - j / NP)) * rj ** 2)
        dvj = np.random.rand(D)
        dvj = dvj if TVj > 0.5 else -dvj

        vj = dvj * Lgs * (Xt[0, :-1] - Xt[j, :-1])

        dvij = np.random.rand(D)
        dvij = dvij if TVj > 0.5 else -dvij
        vij = dvij * (Xt[j, :-1] - Xt[i, :-1])

        xit1_candidate = Xt[i, :-1] + vij + vj
        xit1 = np.clip(xit1_candidate, [b[0] for b in bounds], [b[1] for b in bounds])
        _project_int(xit1, int_dims)
        fx = min_func(xit1)

        Xt1.append(np.append(xit1, fx))

    Xt1 = np.array(Xt1)
    Xt1 = Xt1[Xt1[:, -1].argsort()]
    return Xt1


def BCMO(min_func, bounds, NP, gen, verbose=True, int_dims=None):
    """Run BCMO for ``gen`` generations with population ``NP``.

    Parameters
    ----------
    min_func : callable
        Objective function to minimize. Takes a 1-D array of length ``len(bounds)``.
    bounds : list of (low, high)
        Per-dimension search bounds.
    NP : int
        Population size.
    gen : int
        Number of generations.
    verbose : bool, default True
        Print best / worst fitness per generation.
    int_dims : list of int, optional
        Dimensions constrained to integers (rounded after every clip per paper
        Algorithm 3.1 line 12). For D_BCMO-Mapper pass ``[0]`` so ``n`` stays
        integer throughout the search.

    Returns
    -------
    X : ndarray, shape (NP, D+1)
        Final population sorted by fitness. Last column is fitness.
    history : list of (best, worst)
        Fitness extrema per generation.
    """
    history = []

    X = initial(NP, bounds, min_func, int_dims=int_dims)
    history.append((float(X[0, -1]), float(X[-1, -1])))

    for ii in range(gen):
        if verbose:
            print("gen = ", ii)
        X = algorithm_1(X, bounds, min_func, int_dims=int_dims)
        X = algorithm_2(X, bounds, min_func, int_dims=int_dims)
        if verbose:
            print("gen, best, worst: ", ii, X[0, -1], X[-1, -1])
        history.append((float(X[0, -1]), float(X[-1, -1])))

    return X, history
