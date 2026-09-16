"""Quadrature, sampling, and statistical weight utilities."""
import math
import warnings
import numpy as np
import pandas as pd
from scipy.stats import qmc
from numpy.polynomial.legendre import leggauss

from cavsim2d.constants import *

#: pandas separator for whitespace-delimited node files.
SEP_WHITESPACE = r"\s+"

def stroud(p):
    """
    Stroud-3 method

    Parameters
    ----------
    p: int
        Dimension

    Returns
    -------
    Nodes of quadrature rule in [0,1]**p (column-wise)
    """
    # Stroud-3 method
    #
    # Input parameters:
    #  p   number of dimensions
    # Output parameters:
    #  nodes   nodes of quadrature rule in [0,1]**p (column-wise)
    #

    nodes = np.zeros((p, 2 * p))
    coeff = np.pi / p
    fac = np.sqrt(2 / 3)

    for i in range(2 * p):
        for r in range(int(np.floor(0.5 * p))):
            k = 2 * r
            nodes[k, i] = fac * np.cos((k + 1) * (i + 1) * coeff)
            nodes[k + 1, i] = fac * np.sin((k + 1) * (i + 1) * coeff)

        if 0.5 * p != np.floor(0.5 * p):
            nodes[-1, i] = ((-1) ** (i + 1)) / np.sqrt(3)

    # transform nodes from [-1,+1]**p to [0,1]**p
    nodes = 0.5 * nodes + 0.5

    return nodes


def quad_stroud3(rdim, degree):
    """
    Stroud-3 quadrature in :math:`[0,1]^k`

    .. note::

        Dimensional Threshold Limitation: In practice, the Stroud 3 quadrature rule may be effective in dimensions
        up to around 3 to 6, depending on the specific problem and the function being integrated. Beyond this,
        the accuracy of the rule typically degrades, and higher-order quadrature rules or
        Monte Carlo methods might be more appropriate.

    Parameters
    ----------
    rdim: int
        Dimension of variables
    degree: int
        Degree

    Returns
    -------
    Nodes and corresponding weights
    """
    # data for Stroud-3 quadrature in [0,1]**k
    # nodes and weights
    nodes = stroud(rdim)
    nodestr = 2. * nodes - 1.
    weights = (1 / (2 * rdim)) * np.ones((2 * rdim, 1))

    # evaluation of Legendre polynomials
    bpoly = np.zeros((degree + 1, rdim, 2 * rdim))
    for ll in range(rdim):
        for j in range(2 * rdim):
            bpoly[0, ll, j] = 1
            bpoly[1, ll, j] = nodestr[ll, j]
            for i in range(1, degree):
                bpoly[i + 1, ll, j] = ((2 * (i + 1) - 1) * nodestr[ll, j] * bpoly[i, ll, j] - i * bpoly[
                    i - 1, ll, j]) / (i + 1)

    # standardisation of Legendre polynomials
    for i in range(1, degree + 1):
        bpoly[i, :, :] = bpoly[i, :, :] * np.sqrt(2 * (i + 1) - 1)
    return nodes, weights, bpoly


def c1_leg_monomial_integral(expon):
    if expon < 0:
        error("\n")
        error("C1_LEG_MONOMIAL_INTEGRAL - Fatal error!")
        error("EXPON < 0.")
        raise ValueError("C1_LEG_MONOMIAL_INTEGRAL - Fatal error!")

    if expon % 2 == 1:
        return 0.0

    value = 2.0 / (expon + 1)
    return value


def cn_leg_03_xiu(n):
    o = 2 * n

    x = np.zeros((n, o))
    w = np.zeros(o)

    expon = 0
    volume = c1_leg_monomial_integral(expon)
    volume = volume ** n

    for j in range(1, o + 1):

        i = 0
        for r in range(1, math.floor(n / 2) + 1):
            arg = (2 * r - 1) * j * np.pi / n
            i += 1
            x[i - 1, j - 1] = np.sqrt(2.0) * np.cos(arg) / np.sqrt(3.0)
            i += 1
            x[i - 1, j - 1] = np.sqrt(2.0) * np.sin(arg) / np.sqrt(3.0)

        if i < n:
            i += 1
            x[i - 1, j - 1] = np.sqrt(2.0) * (-1) ** j / np.sqrt(3.0)
            if n == 1:
                x[i - 1, j - 1] = x[i - 1, j - 1] / np.sqrt(2.0)

    w[0:o] = volume / o

    return x, np.atleast_2d(w).T / np.sum(w)


def cn_leg_03_1(n):
    o = 2 * n

    w = np.zeros(o)
    x = np.zeros((n, o))

    expon = 0
    volume = c1_leg_monomial_integral(expon)
    volume = volume ** n

    for j in range(1, o + 1):

        i = 0

        for r in range(1, math.floor(n / 2) + 1):
            arg = (2 * r - 1) * j * np.pi / n
            i += 1
            x[i - 1, j - 1] = np.sqrt(2.0) * np.cos(arg) / np.sqrt(3.0)
            i += 1
            x[i - 1, j - 1] = np.sqrt(2.0) * np.sin(arg) / np.sqrt(3.0)

        if i < n:
            i += 1
            if n == 1:
                x[i - 1, j - 1] = r8_mop(j) / np.sqrt(3.0)
            else:
                x[i - 1, j - 1] = np.sqrt(2.0) * r8_mop(j) / np.sqrt(3.0)

    w[0:o] = volume / o

    return x, np.atleast_2d(w).T / np.sum(w)


def r8_mop(i):
    if i % 2 == 0:
        value = 1.0
    else:
        value = -1.0

    return value


def cn_leg_05_1(n, option=1):
    """
    The rule has order

    O = 2 N^2 + N + 2.

    The rule has precision P = 5.

    CN_LEG is the cube [-1,+1]^N with the Legendre weight function

    w(x) = 1.

    .. note::

        Dimensional Threshold Limitation: In practice, the Stroud 3 quadrature rule may be effective in dimensions
        up to around 3 to 6, depending on the specific problem and the function being integrated. Beyond this,
        the accuracy of the rule typically degrades, and higher-order quadrature rules or
        Monte Carlo methods might be more appropriate.

    Parameters
    ----------
    n
    option

    Returns
    -------

    """
    # Check if the value of n is 4, 5, or 6
    if n not in [4, 5, 6]:
        error("\n")
        error("CN_LEG_05_1 - Fatal error!")
        error("The value of N must be 4, 5, or 6.")
        raise ValueError("CN_LEG_05_1 - Fatal error!")

    # Check for valid option when n = 4 or 5
    if n in [4, 5] and option not in [1, 2]:
        error("\n")
        error("CN_LEG_05_1 - Fatal error!")
        error("When N = 4 or 5, OPTION must be 1 or 2.")
        raise ValueError("CN_LEG_05_1 - Fatal error!")

    o = n ** 2 + n + 2
    w = np.zeros(o)
    x = np.zeros((n, o))

    expon = 0
    volume = c1_leg_monomial_integral(expon)
    volume = volume ** n

    if (n == 4 and option == 1):
        eta = 0.778984505799815
        lmbda = 1.284565137874656
        xsi = -0.713647298819253
        mu = -0.715669761974162
        gamma = 0.217089151000943
        a = 0.206186096875899e-01 * volume
        b = 0.975705820221664e-02 * volume
        c = 0.733921929172573e-01 * volume
    elif (n == 4 and option == 2):
        eta = 0.546190755827425E+00
        lmbda = 0.745069130115661E+00
        xsi = - 0.413927294508700E+00
        mu = - 0.343989637454535E+00
        gamma = 1.134017894600344E+00
        a = 0.853094758323323E-01 * volume
        b = 0.862099000096395E-01 * volume
        c = 0.116418206881849E-01 * volume
    elif (n == 5 and option == 1):
        eta = 0.522478547481276E+00
        lmbda = 0.936135175985774E+00
        xsi = - 0.246351362101519E+00
        mu = - 0.496308106093758E+00
        gamma = 0.827180176822930E+00
        a = 0.631976901960153E-01 * volume
        b = 0.511464127430166E-01 * volume
        c = 0.181070246088902E-01 * volume
    elif (n == 5 and option == 2):
        eta = 0.798317301388741E+00
        lmbda = 0.637344273885728E+00
        xsi = - 0.455245909918377E+00
        mu = - 1.063446229997311E+00
        gamma = 0.354482076665770E+00
        a = 0.116952384292206E-01 * volume
        b = 0.701731258612708E-01 * volume
        c = 0.137439132264426E-01 * volume
    else:
        eta = 0.660225291773525E+00
        lmbda = 1.064581294844754E+00
        xsi = 0.000000000000000E+00
        mu = - 0.660225291773525E+00
        gamma = 0.660225291773525E+00
        a = 0.182742214532872E-01 * volume
        b = 0.346020761245675E-01 * volume
        c = 0.182742214532872E-01 * volume

    # Set x and w based on parameters
    k = 0
    # k += 1
    for i in range(n):
        x[i, k] = eta
    w[k] = a

    # k += 1
    for i in range(n):
        x[i, k] = -eta
    w[k] = a

    for i1 in range(n):
        for i in range(1, n):
            x[i, k] = xsi
        x[i1, k] = lmbda
        w[k] = b
        k = k + 1

    for i1 in range(n):
        for i in range(n):
            x[i, k] = - xsi
        x[i1, k] = - lmbda
        w[k] = b
        k = k + 1

    for i1 in range(n - 1):
        for i2 in range(i1 + 1, n):
            for i in range(n):
                x[i, k] = gamma
            x[i1, k] = mu
            x[i2, k] = mu
            w[k] = c
            k = k + 1

    for i1 in range(n - 1):
        for i2 in range(i1 + 1, n):
            for i in range(n):
                x[i, k] = - gamma
            x[i1, k] = - mu
            x[i2, k] = - mu
            w[k] = c
            k = k + 1

    return x, np.atleast_2d(w).T / np.sum(w)


def cn_leg_05_2(n):
    """
    The rule has order

    O = 2 N^2 + 1.

    The rule has precision P = 5.

    CN_LEG is the cube [-1,+1]^N with the Legendre weight function

    w(x) = 1.

    .. note::

        Dimensional Threshold Limitation: In practice, the Stroud 5 quadrature rule may be effective in dimensions
        up to around 5 to 10, depending on the specific problem and the function being integrated. Beyond this,
        the accuracy of the rule typically degrades, and higher-order quadrature rules or
        Monte Carlo methods might be more appropriate.

    Parameters
    ----------
    n
    option

    Returns
    -------

    """
    if n < 2:
        error("CN_LEG_05_2 - Fatal error!")
        error("N must be at least 2.")
        raise ValueError("CN_LEG_05_2 - Fatal error!")

    o = 2 * n ** 2 + 1
    w = np.zeros(o, dtype=np.float64)
    x = np.zeros((n, o), dtype=np.float64)

    expon = 0
    volume = c1_leg_monomial_integral(expon)
    volume = volume ** n

    b0 = (25 * n * n - 115 * n + 162) * volume / 162.0
    b1 = (70 - 25 * n) * volume / 162.0
    b2 = 25.0 * volume / 324.0

    r = np.sqrt(3.0 / 5.0)

    k = 0

    k += 1
    for i in range(n):
        x[i, k - 1] = 0.0
    w[k - 1] = b0

    for i1 in range(1, n + 1):
        k += 1
        for i in range(n):
            x[i, k - 1] = 0.0
        x[i1 - 1, k - 1] = +r
        w[k - 1] = b1

        k += 1
        for i in range(n):
            x[i, k - 1] = 0.0
        x[i1 - 1, k - 1] = -r
        w[k - 1] = b1

    for i1 in range(1, n):
        for i2 in range(i1 + 1, n + 1):
            k += 1
            for i in range(n):
                x[i, k - 1] = 0.0
            x[i1 - 1, k - 1] = +r
            x[i2 - 1, k - 1] = +r
            w[k - 1] = b2

            k += 1
            for i in range(n):
                x[i, k - 1] = 0.0
            x[i1 - 1, k - 1] = +r
            x[i2 - 1, k - 1] = -r
            w[k - 1] = b2

            k += 1
            for i in range(n):
                x[i, k - 1] = 0.0
            x[i1 - 1, k - 1] = -r
            x[i2 - 1, k - 1] = +r
            w[k - 1] = b2

            k += 1
            for i in range(n):
                x[i, k - 1] = 0.0
            x[i1 - 1, k - 1] = -r
            x[i2 - 1, k - 1] = -r
            w[k - 1] = b2

    return x, np.atleast_2d(w).T / np.sum(w)


def cn_gauss(rdim, degree):
    x, w = np.polynomial.legendre.leggauss(degree)

    X = [x for _ in range(rdim)]

    nodes = np.array(np.meshgrid(*X, indexing='ij')).reshape(rdim, -1)
    weights = np.ones(degree ** rdim) / (degree ** rdim)

    return nodes, np.atleast_2d(weights).T / np.sum(weights)


def weighted_mean_obj(tab_var, weights):
    # print(weights, weights.shape)
    # print(tab_var, tab_var.shape)
    rows_sims_no, cols = np.shape(tab_var)
    no_weights, dummy = np.shape(weights)
    if rows_sims_no == no_weights:
        # expe = np.zeros((cols, 1))
        # outvar = np.zeros((cols, 1))
        # for i in range(cols):
        #     expe[i, 0] = np.dot(tab_var[:, i], weights)[0]
        #     outvar[i, 0] = np.dot(tab_var[:, i] ** 2, weights)[0]
        #
        # stdDev = np.sqrt(abs(outvar - expe ** 2))
        mean = weighted_mean(tab_var, weights.T[0])
        var = weighted_variance(tab_var, weights.T[0], mean)
        # A negative variance is only possible with a negative-weight rule
        # (Stroud5 at k >= 3). np.sqrt would turn it into a silent NaN, so say what
        # happened: the moments from such a rule cannot be trusted for this
        # response, rather than the solver having failed.
        if np.any(np.asarray(var) < 0):
            bad = int(np.sum(np.asarray(var) < 0))
            warnings.warn(
                f"UQ: {bad} of {np.size(var)} objectives came out with a NEGATIVE "
                f"weighted variance, so their stdDev is NaN. This means the node "
                f"rule has negative weights (Stroud5 at 3+ variables) and the "
                f"response varies too sharply across its axis nodes for the rule to "
                f"hold. Re-run those objectives with a Monte-Carlo design.",
                UserWarning, stacklevel=3)
        std = np.sqrt(np.asarray(var, dtype=float))
        skew = weighted_skew(tab_var, weights.T[0], mean, std)
        kurtosis = weighted_kurtosis(tab_var, weights.T[0], mean, std)
    else:
        mean = 0
        std = 0
        skew = 0
        kurtosis = 0
        error('Cols_sims_no != No_weights')
    return list(mean), list(std), list(skew), list(kurtosis)


def weighted_mean(var, wts):
    """Calculates the weighted mean"""
    return np.average(var, weights=wts, axis=0)


def weighted_variance(var, wts, mean):
    """Calculates the weighted variance"""
    return np.average((var - mean) ** 2, weights=wts, axis=0)


def weighted_skew(var, wts, mean, std):
    """Calculates the weighted skewness, returning NaN where std==0."""
    num = np.average((var - mean) ** 3, weights=wts, axis=0)
    denom = std**3
    # allocate output, fill with NaN
    skew = np.full_like(num, np.nan, dtype=float)
    # divide only where denom != 0
    np.divide(num, denom, out=skew, where=(denom != 0))
    return skew


def weighted_kurtosis(var, wts, mean, std):
    """Calculates the weighted kurtosis, returning NaN where std==0."""
    num = np.average((var - mean) ** 4, weights=wts, axis=0)
    denom = std**4
    kurt = np.full_like(num, np.nan, dtype=float)
    np.divide(num, denom, out=kurt, where=(denom != 0))
    return kurt


def normal_dist(x, mean, sd):
    prob_density = (np.pi * sd) * np.exp(-0.5 * ((x - mean) / sd) ** 2)
    return prob_density



def stroud3_nodes_and_weights(p: int):
    """Stroud’s 3rd-degree rule nodes & weights in [0,1]^p."""
    coeff = np.pi / p
    fac = np.sqrt(2 / 3)
    raw = np.zeros((p, 2 * p))
    for i in range(2 * p):
        for r in range(p // 2):
            k = 2 * r
            raw[k, i] = fac * np.cos((k + 1) * (i + 1) * coeff)
            raw[k + 1, i] = fac * np.sin((k + 1) * (i + 1) * coeff)
        if p % 2:
            raw[-1, i] = ((-1) ** (i + 1)) / np.sqrt(3)
    nodes = (0.5 * raw + 0.5).T  # (2p, p)
    weights = np.full(2 * p, 1.0 / (2 * p))
    return nodes, weights


def _bound_vector(k: int, bound):
    """*bound* as a length-*k* array.

    ``perturb_geometry`` always passes one delta PER expanded variable, so every
    generator has to broadcast per-dimension. Taking a scalar too keeps the
    generators usable directly.
    """
    b = np.asarray(bound, dtype=float)
    if b.ndim == 0:
        return np.full(k, float(b))
    if b.size != k:
        raise ValueError(f"expected {k} perturbation bounds (one per variable), got {b.size}")
    return b.reshape(k)


def _require_n(n, method):
    """Sample count for the rules that have no intrinsic one."""
    if n is None:
        raise ValueError(
            f"the {method!r} node rule needs a sample count: pass it as the second "
            f"element of 'method', e.g. ['{method}', 50].")
    n = int(n)
    if n < 2:
        raise ValueError(f"{method!r} needs at least 2 samples, got {n}.")
    return n


def generate_uniform_nodes(k: int, bound, n: int):
    """Uniform random delta-vectors in [-bound,+bound] with equal weights."""
    n = _require_n(n, 'Uniform')
    b = _bound_vector(k, bound)
    # Via np.asarray, so a per-variable LIST of deltas negates elementwise
    # instead of raising "bad operand type for unary -: 'list'".
    deltas = [np.random.uniform(-b, b, size=k) for _ in range(n)]
    weights = [1.0 / n] * n
    return deltas, weights


def generate_normal_nodes(k: int, bound, n: int, seed=None):
    """
    n independent multivariate normal samples in k dims,
    each component ~ N(0,bound^2).
    """
    n = _require_n(n, 'Normal')
    b = _bound_vector(k, bound)
    rng     = np.random.default_rng(seed)
    sample  = rng.standard_normal(size=(n, k)) * b
    deltas  = list(sample)
    weights = [1.0/n]*n
    return deltas, weights


def generate_gauss_legendre_nodes(k: int, bound, n: int):
    """Tensor-product Gauss–Legendre nodes & weights on [-bound,bound].

    n**k nodes, so this is affordable for a handful of variables only.
    """
    n = _require_n(n, 'Gauss_Legendre')
    b = _bound_vector(k, bound)
    x1d, w1d = leggauss(n)
    grids = np.meshgrid(*([x1d] * k), indexing='ij')
    wgrids = np.meshgrid(*([w1d] * k), indexing='ij')
    # Scale each DIMENSION by its own bound after the tensor product. Scaling the
    # 1-D rule first ("x1d *= bound") broadcast a length-n array against a
    # length-k one, so it raised for every k != n.
    flat_x = np.stack([g.ravel() for g in grids], axis=1) * b   # (n**k, k)
    flat_w = np.prod([wg.ravel() for wg in wgrids], axis=0)     # (n**k,)
    flat_w = flat_w / flat_w.sum()
    return list(flat_x), list(flat_w)


def generate_stroud3_nodes(k: int, bound):
    """Stroud-III delta-vectors mapped to [-bound,bound] and equal weights."""
    b = _bound_vector(k, bound)
    nodes, w = stroud3_nodes_and_weights(k)
    deltas = [(vec - 0.5) * 2 * b for vec in nodes]
    return deltas, list(w)


def generate_stroud5_nodes(k: int, bound):
    """Stroud-5 (degree-5) delta-vectors on [-bound,bound].

    2k^2+1 nodes against Stroud-3's 2k. It integrates quartics exactly, so it is
    the cheapest independent check on a Stroud-3 result.
    """
    b = _bound_vector(k, bound)
    nodes, w = cn_leg_05_2(k)      # (k, N) on [-1,1]^k; weights (N, 1) summing to 1
    deltas = [vec * b for vec in np.asarray(nodes).T]
    return deltas, list(np.asarray(w).ravel())


def generate_lhs_nodes(k: int, bound, n: int, seed=None):
    """Latin-hypercube delta-vectors in [-bound,+bound] with equal weights."""
    n = _require_n(n, 'LHS')
    b = _bound_vector(k, bound)
    sample = qmc.LatinHypercube(d=k, seed=seed).random(n)       # (n, k) in [0, 1]
    deltas = list((2.0 * sample - 1.0) * b)
    weights = [1.0 / n] * n
    return deltas, weights


def generate_nodes_from_file(k: int, bound, path):
    """Delta-vectors read from a whitespace-separated file, one node per row.

    The file holds the perturbations themselves (same units as ``delta``), so
    *bound* is not applied. Equal weights.
    """
    if path is None:
        raise ValueError(
            "the 'from file' node rule needs a path: ['from file', '<path>'].")
    arr = pd.read_csv(path, sep=SEP_WHITESPACE).to_numpy(dtype=float)
    if arr.ndim != 2 or arr.shape[1] != k:
        got = arr.shape[1] if arr.ndim == 2 else '?'
        raise ValueError(
            f"{path!r} has {got} columns but there are {k} perturbed variables "
            f"— one column per variable is required.")
    n = arr.shape[0]
    if n < 2:
        raise ValueError(f"{path!r} holds {n} node(s); at least 2 are needed.")
    return list(arr), [1.0 / n] * n


#: Node rules reachable through :func:`generate_nodes`, for error messages.
NODE_RULES = ('Stroud3', 'Stroud5', 'Uniform', 'Normal', 'LHS',
              'Gauss_Legendre', 'from file')


def generate_nodes(k: int, bound, node_type: list, seed=3799):
    """Dispatch to the appropriate node generator.

    *node_type* is ``[rule]`` or ``[rule, parameter]``, the parameter being a
    sample count ('Uniform', 'Normal', 'LHS', 'Gauss_Legendre') or a path
    ('from file'). ``['Quadrature', '<rule>']`` is accepted too, which is how
    the documented default ``['Quadrature', 'Stroud3']`` is spelled.
    """
    if not node_type:
        raise ValueError(f"'method' is empty; choose one of {', '.join(NODE_RULES)}.")
    method_name = str(node_type[0]).lower()
    params = node_type[1] if len(node_type) > 1 else None

    # ['Quadrature', 'Stroud3'] -> the second element names the algorithm, so a
    # third would carry its parameter. Bare ['Quadrature'] names no rule and
    # falls through to the error below rather than silently picking one.
    if method_name == 'quadrature' and params is not None:
        method_name = str(params).lower()
        params = node_type[2] if len(node_type) > 2 else None

    check_node_rule(k, node_type)

    if method_name == 'uniform':
        return generate_uniform_nodes(k, bound, params)
    elif method_name == 'normal':
        return generate_normal_nodes(k, bound, params, seed=seed)
    elif method_name == 'lhs':
        return generate_lhs_nodes(k, bound, params, seed=seed)
    elif method_name == 'gauss_legendre':
        return generate_gauss_legendre_nodes(k, bound, params)
    elif method_name == 'stroud3':
        return generate_stroud3_nodes(k, bound)
    elif method_name == 'stroud5':
        return generate_stroud5_nodes(k, bound)
    elif method_name in ('from file', 'from_file', 'file'):
        return generate_nodes_from_file(k, bound, params)

    raise ValueError(
        f"Unknown UQ node rule {node_type[0]!r}"
        + (f" / {node_type[1]!r}" if len(node_type) > 1 else "")
        + f". Choose one of: {', '.join(NODE_RULES)}.")


# ---------------------------------------------------------------------------
# Rule adequacy
# ---------------------------------------------------------------------------

#: Dimension up to which each cubature rule's moments can be trusted WITHOUT an
#: independent cross-check. These are not cliffs: measured against exact answers,
#: Stroud3's sigma error on a smooth weakly-nonlinear response stays ~1-2% even at
#: k=20. The limit is that a 2k-node degree-3 design cannot itself reveal how much
#: degree->=4 content the response has, so past this width the result needs a
#: second rule to confirm it.
RULE_DIM_COMFORT = {'stroud3': 6, 'stroud5': 12}

#: Node budget above which a tensor-product rule is refused as impractical.
TENSOR_NODE_BUDGET = 20000

#: Relative standard error on a reported stdDev that triggers a sample-count
#: warning. sigma's own relative standard error is ~1/sqrt(2N), independent of
#: dimension (verified against repeated sampling).
SIGMA_RSE_WARN = 0.10


def sigma_relative_standard_error(n):
    """Relative standard error of a stdDev estimated from *n* iid samples.

    ``1/sqrt(2n)`` — dimension-independent, which is what separates UQ sample
    sizing from Saltelli sensitivity sizing (the latter scales with the number of
    variables, this does not).
    """
    return 1.0 / np.sqrt(2.0 * int(n))


def samples_for_sigma_accuracy(rel_err):
    """Samples needed for a stdDev accurate to *rel_err* (e.g. 0.05 -> 5%)."""
    return int(np.ceil(1.0 / (2.0 * float(rel_err) ** 2)))


def check_node_rule(k, node_type, n_nodes=None, stacklevel=4):
    """Warn when a UQ node rule is used where its result cannot be trusted.

    Emits ``UserWarning``s rather than raising: the run is still meaningful, but
    the caller needs to know the moments are unvalidated. Raises only for a
    tensor rule whose node count is beyond any practical solver budget.
    """
    if not node_type:
        return
    rule = str(node_type[0]).lower()
    params = node_type[1] if len(node_type) > 1 else None
    if rule == 'quadrature' and params is not None:
        rule = str(params).lower()
        params = node_type[2] if len(node_type) > 2 else None

    comfort = RULE_DIM_COMFORT.get(rule)
    if comfort is not None and k > comfort:
        better = "Stroud5" if rule == 'stroud3' else "a Monte-Carlo design ('LHS'/'Normal')"
        warnings.warn(
            f"UQ: {rule!r} over {k} random variables is past the {comfort} "
            f"dimensions where its moments can be trusted unchecked. It stays exact "
            f"for degree-{3 if rule == 'stroud3' else 5} responses at any width, but "
            f"{2 * k if rule == 'stroud3' else 2 * k * k + 1} nodes cannot reveal how "
            f"much higher-order content this response has. Cross-check with {better}, "
            f"or treat the stdDev as indicative.",
            UserWarning, stacklevel=stacklevel)

    if rule == 'stroud5' and k >= 3:
        # cn_leg_05_2 puts NEGATIVE weights on the 2k axis nodes for every k >= 3
        # (min weight -0.03 at k=3, falling linearly to -1.42 at k=12). The rule is
        # still degree-5 exact, but the quadrature is no longer a probability
        # measure: if the response varies strongly across those nodes the weighted
        # variance can come out NEGATIVE and the stdDev becomes NaN.
        w_min = -(k - 2) * 0.154321 + 0.077160
        warnings.warn(
            f"UQ: 'Stroud5' at {k} variables uses negative weights on its axis nodes "
            f"(min ~{w_min:.2f}), so it is not a probability measure. It stays "
            f"degree-5 exact for smooth responses, but a sharply varying one can "
            f"produce a negative variance and a NaN stdDev. Check the reported "
            f"stdDev is finite, and prefer a Monte-Carlo design if it is not.",
            UserWarning, stacklevel=stacklevel)

    if rule == 'gauss_legendre' and params is not None:
        total = int(params) ** k
        if total > TENSOR_NODE_BUDGET:
            raise ValueError(
                f"UQ: ['Gauss_Legendre', {params}] over {k} variables is {params}**{k} "
                f"= {total:,} solver runs, beyond the {TENSOR_NODE_BUDGET:,} budget. A "
                f"tensor rule is only affordable for a handful of variables — use "
                f"'Stroud5' (degree 5 at {2 * k * k + 1} nodes) instead.")
        warnings.warn(
            f"UQ: ['Gauss_Legendre', {params}] over {k} variables is {total:,} solver "
            f"runs. 'Stroud5' reaches degree 5 in {2 * k * k + 1}.",
            UserWarning, stacklevel=stacklevel)

    if rule in ('uniform', 'normal', 'lhs') and params is not None:
        n = int(params)
        rse = sigma_relative_standard_error(n)
        if rse > SIGMA_RSE_WARN:
            warnings.warn(
                f"UQ: {n} samples gives a stdDev with ~{rse * 100:.0f}% relative "
                f"standard error (1/sqrt(2N)). For {SIGMA_RSE_WARN * 100:.0f}% use "
                f"N>={samples_for_sigma_accuracy(SIGMA_RSE_WARN)}, for 5% use "
                f"N>={samples_for_sigma_accuracy(0.05)}, for 1% use "
                f"N>={samples_for_sigma_accuracy(0.01)}. This does not depend on the "
                f"number of variables.",
                UserWarning, stacklevel=stacklevel)
        if rule == 'lhs':
            warnings.warn(
                "UQ: Latin hypercube sharpens the MEAN (often by 1-3 orders of "
                "magnitude) but not the stdDev — sigma's error still follows "
                "1/sqrt(2N). Size N from the stdDev you need.",
                UserWarning, stacklevel=stacklevel)


def bootstrap_moment_errors(tab_var, weights, n_boot=2000, seed=12345):
    """Standard errors of the weighted mean and stdDev, by bootstrap.

    Returns ``(se_mean, se_std)``, one entry per column of *tab_var*. Valid only
    for an **iid sample** design ('Uniform', 'Normal', 'LHS'): a cubature rule's
    nodes are placed deterministically, so resampling them estimates nothing and
    the caller must not use this for one.

    This is the convergence check a single UQ run can actually afford — it reuses
    the solves already done and costs no extra runs.
    """
    tab = np.asarray(tab_var, dtype=float)
    w = np.asarray(weights, dtype=float).reshape(-1)
    n = tab.shape[0]
    if n < 3:
        return ([np.nan] * tab.shape[1], [np.nan] * tab.shape[1])
    rng = np.random.default_rng(seed)
    means = np.empty((n_boot, tab.shape[1]))
    stds = np.empty((n_boot, tab.shape[1]))
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        wb = w[idx]
        m = np.average(tab[idx], weights=wb, axis=0)
        v = np.average((tab[idx] - m) ** 2, weights=wb, axis=0)
        means[b] = m
        stds[b] = np.sqrt(v)
    return list(np.nanstd(means, axis=0)), list(np.nanstd(stds, axis=0))


def is_sampling_rule(node_type):
    """True when *node_type* draws an iid sample (so a bootstrap is meaningful)."""
    if not node_type:
        return False
    rule = str(node_type[0]).lower()
    if rule == 'quadrature' and len(node_type) > 1:
        rule = str(node_type[1]).lower()
    return rule in ('uniform', 'normal', 'lhs')
