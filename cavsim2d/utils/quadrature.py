"""Quadrature, sampling, and statistical weight utilities."""
import numpy as np
from scipy.stats import qmc
from numpy.polynomial.legendre import leggauss

from cavsim2d.constants import *

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
        std = np.sqrt(weighted_variance(tab_var, weights.T[0], mean))
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


def generate_uniform_nodes(k: int, bound: float, n: int):
    """Uniform random delta-vectors in [-bound,+bound] with equal weights."""
    deltas = [np.random.uniform(-bound, bound, size=k) for _ in range(n)]
    weights = [1.0 / n] * n
    return deltas, weights

def generate_normal_nodes(k: int, bound: float, n: int, seed=None):
    """
    n independent multivariate normal samples in k dims,
    each component ~ N(0,bound^2).
    """
    rng     = np.random.default_rng(seed)
    sample  = rng.standard_normal(size=(n, k)) * bound
    deltas  = list(sample)
    weights = [1.0/n]*n
    return deltas, weights

def generate_gauss_legendre_nodes(k: int, bound: float, n: int):
    """Tensor-product Gauss–Legendre nodes & weights on [-bound,bound]."""
    x1d, w1d = leggauss(n)
    x1d *= bound
    w1d *= bound
    grids = np.meshgrid(*([x1d] * k), indexing='ij')
    wgrids = np.meshgrid(*([w1d] * k), indexing='ij')
    flat_x = np.stack([g.ravel() for g in grids], axis=1)  # (n**k, k)
    flat_w = np.prod([wg.ravel() for wg in wgrids], axis=0)  # (n**k,)
    return list(flat_x), list(flat_w)


def generate_stroud3_nodes(k: int, bound: float):
    """Stroud-III delta-vectors mapped to [-bound,bound] and equal weights."""
    nodes, w = stroud3_nodes_and_weights(k)
    deltas = [(vec - 0.5) * 2 * bound for vec in nodes]
    return deltas, list(w)


def generate_nodes(k: int, bound: float, node_type: list):
    """Dispatch to the appropriate node generator."""
    # Flatten or identify method name from ['Category', 'Method']
    method_name = node_type[0].lower()
    params = node_type[1] if len(node_type) > 1 else None
    
    # If the first element is 'quadrature', look at the second for the algorithm
    if method_name == 'quadrature' and params:
        method_name = params.lower()

    if method_name == 'uniform':
        return generate_uniform_nodes(k, bound, params)
    elif method_name == 'normal':
        return generate_normal_nodes(k, bound, params, seed=3799)
    elif method_name == 'gauss_legendre':
        return generate_gauss_legendre_nodes(k, bound, params)
    elif method_name == 'stroud3':
        return generate_stroud3_nodes(k, bound)
    
    raise ValueError(f"Unknown node_type {method_name!r}")


