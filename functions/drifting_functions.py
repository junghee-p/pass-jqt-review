"""
Test functions and the abrupt mean-drift generator.

Provides `get_y_value`, which evaluates a test function at a batch of inputs
and optionally injects an abrupt mean drift (additive shift) inside a
localized drift region.
"""
import numpy as np


# =============================================================================
# Top-level entry point
# =============================================================================
def get_y_value(input, noise, function, drift_magnitude=0, drift_position=None,
                drift_function='partial', drift_w=0, pure=False, rng=None):
    """
    Evaluate `function` on `input` and add Gaussian observation noise.

    Parameters
    ----------
    input : (n, d) array
        Input points.
    noise : float
        Observation noise standard deviation sigma.
    function : str
        One of the function names defined below.
    drift_magnitude : float
        Delta in units of sigma. The additive shift inside the drift region is
        Delta * sigma.
    drift_position : (d,) array or None
        Center of the drift region.
    drift_function : str
        Drift-region shape; currently only 'partial' (a hyper-rectangle).
    drift_w : float
        Half-width of the drift region along each axis.
    pure : bool
        If True, return the noiseless function value.
    rng : np.random.Generator or None
        PRNG for noise generation.
    """
    if rng is None:
        rng = np.random.default_rng(rng)

    # Dispatch table of test functions
    func_map = {
        'branin_2d':     branin_2d,
        'ishigami_3d':   ishigami_3d,
        'friedman_5d':   friedman_5d,
        'linkletter_8d': linkletter_8d,
        'welch_20d':     welch_20d,
    }
    if function not in func_map:
        raise ValueError(f"Unknown function: {function}")

    y_value = func_map[function](input)

    # Abrupt mean drift: additive shift inside the drift region.
    if drift_magnitude > 0:
        anomalies = _add_anomalies(drift_function, input, drift_position,
                                   drift_magnitude, noise, drift_w)
        y_value = y_value + anomalies

    if pure:
        return y_value
    return y_value + rng.normal(0, noise, len(input))


# =============================================================================
# Drift region geometry
# =============================================================================
def _drift_region_indicator(drift_function, points, drift_position, drift_w):
    """Boolean mask: True for points inside the drift region."""
    if drift_function == 'partial':
        return np.all(np.abs(points - drift_position) <= drift_w, axis=1)
    raise ValueError(f"Unknown drift_function: {drift_function}")


def _add_anomalies(drift_function, input, drift_position, drift_magnitude, noise, drift_w):
    """Per-point additive drift term `Delta * sigma * 1{in drift region}`."""
    anomaly = _drift_region_indicator(drift_function, input, drift_position, drift_w)
    return anomaly * drift_magnitude * noise if np.any(anomaly) else 0


# =============================================================================
# Test functions
# =============================================================================
def branin_2d(x):
    """Branin function, d=2. Domain: x1 in [-5,10], x2 in [0,15]."""
    a, b, c, r, s, t = 1, 5.1 / (4 * np.pi ** 2), 5 / np.pi, 6, 10, 1 / (8 * np.pi)
    term1 = a * (x[:, 1] - b * x[:, 0] ** 2 + c * x[:, 0] - r) ** 2
    term2 = s * (1 - t) * np.cos(x[:, 0])
    return term1 + term2 + s


def ishigami_3d(x, a=7, b=0.1):
    """Ishigami function, d=3. Domain: x_i in [-pi, pi]."""
    return (np.sin(x[:, 0])
            + a * np.sin(x[:, 1]) ** 2
            + b * x[:, 2] ** 4 * np.sin(x[:, 0]))


def friedman_5d(x):
    """Friedman function, d=5. Input expected in [0, 10] per dim."""
    x = x / 10  # rescale to [0, 1] for the canonical Friedman form
    return (10 * np.sin(np.pi * x[:, 0] * x[:, 1])
            + 20 * (x[:, 2] - 0.5) ** 2
            + 10 * x[:, 3]
            + 5 * x[:, 4])


def linkletter_8d(x):
    """Linkletter decay function, d=8. Domain: x_i in [0, 1]."""
    coeffs = np.array([2 ** -i for i in range(8)])
    return np.sum(x * 0.2 * coeffs, axis=1)


def welch_20d(x):
    """
    Welch et al. (1992) function, d=20. Domain: x_i in [-0.5, 0.5].
    Mix of strong, weak, and inactive inputs; used for the scalability study.
    """
    return (5.0 * x[:, 11] / (1.0 + x[:, 0])
            + 5.0 * (x[:, 3] - x[:, 19]) ** 2
            + x[:, 4]
            + 40.0 * x[:, 18] ** 3 - 5.0 * x[:, 18]
            + 0.05 * x[:, 1] + 0.08 * x[:, 2]
            - 0.03 * x[:, 5] + 0.03 * x[:, 6]
            - 0.09 * x[:, 8] - 0.01 * x[:, 9] - 0.07 * x[:, 10]
            + 0.25 * x[:, 12] ** 2
            - 0.04 * x[:, 13] + 0.06 * x[:, 14]
            - 0.01 * x[:, 16] - 0.03 * x[:, 17])
