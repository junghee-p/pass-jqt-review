"""
Initialization utilities.

- `model_initialization`: load a pre-fit predictor from disk and generate a
  fresh uniform random reference sample used to estimate sigma_hat and the
  in-control target of the monitoring statistic.
- `monitoring_initialization`: compute sigma_hat, set the in-control target of
  the chart, and attach the user-supplied UCL.
"""
import numpy as np
import joblib
from scipy.special import polygamma
from functions.drifting_functions import get_y_value


# =============================================================================
# Model + initial reference sample
# =============================================================================
def model_initialization(function, boundaries, n_init_sample, noise,
                         predictor_path, seed=0):
    """
    Load the predictor from `predictor_path` and generate an initial
    sample (x_initial, y_initial) used for the predictor.

    The returned reference sample is used downstream to:
      (i)  estimate sigma_hat from the reference residuals, and
      (ii) seed the available-pool at the start.
    """
    rng = np.random.default_rng(seed)
    prediction_model = joblib.load(predictor_path)

    # Uniform random initial sample.
    x_initial = np.column_stack([rng.uniform(b[0], b[1], n_init_sample) for b in boundaries])
    y_initial = get_y_value(x_initial, noise, function, rng=rng)

    return prediction_model, x_initial, y_initial


# =============================================================================
# Monitoring initialization with a user-supplied UCL
# =============================================================================
def monitoring_initialization(mon, prediction_model, x_initial, y_initial,
                              n_init_sample, budget, UCL, seed=0):
    """
    Compute sigma_hat from the reference residuals and build the chart dict.

    Parameters
    ----------
    mon : {'variance', 'average'}
        'variance' : log sample-variance chart V_t.
        'average'  : top-r absolute-residual mean chart A_t.
    UCL : float
        Calibrated upper control limit (targeting ARL0 = 200). Must be supplied
        by the caller; no internal calibration is performed here.
    """
    n_ref = len(x_initial)

    # sigma_hat from the reference residuals, with d.o.f. correction based on
    # the predictor's effective feature count when it exposes one.
    res = y_initial - prediction_model.predict(x_initial)
    rss = float(np.dot(res, res))
    p = int(getattr(prediction_model, '_n_features_out', 0))
    df = (n_ref - p) if (p > 0 and p < n_ref) else n_ref
    sigma_hat = np.sqrt(rss / df)

    monitoring_dic = {}
    if mon == 'average':
        r_largest = budget // 3
        target_a, _ = _estimate_abs_topr_target_sigma(sigma_hat, budget, r_largest, seed=seed)
        monitoring_dic['average'] = {
            'target_a': target_a, 'UCL': UCL, 'r_largest': r_largest,
        }
    elif mon == 'variance':
        target_v, _ = _variance_log_target_sigma(sigma_hat, budget)
        monitoring_dic['variance'] = {'target_v': target_v, 'UCL': UCL}
    else:
        raise ValueError(f"Unknown monitoring type: {mon}")

    return monitoring_dic, prediction_model, sigma_hat


# =============================================================================
# In-control target estimators
# =============================================================================
def _estimate_abs_topr_target_sigma(sigma, budget, r, trials=10000, seed=0):
    """Monte-Carlo mean and std of the top-r average |e_i| for e_i ~ N(0, sigma^2)."""
    rng = np.random.default_rng(seed)
    res = rng.normal(0.0, sigma, size=(trials, budget))
    vals = np.partition(np.abs(res), -r, axis=1)[:, -r:].mean(axis=1)
    return float(vals.mean()), float(vals.std(ddof=1))


def _variance_log_target_sigma(sigma, m):
    """Closed-form mean and std of log(s^2) for s^2 from m N(0, sigma^2) samples."""
    target = (np.log(sigma ** 2)
              + polygamma(0, (m - 1) / 2.0)
              + np.log(2.0)
              - np.log(m - 1.0))
    var = polygamma(1, (m - 1) / 2.0)
    return float(target), float(np.sqrt(var))
