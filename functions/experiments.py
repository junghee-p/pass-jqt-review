"""
Run a single monitoring experiment: PASS (or Random) sampling, followed by
the EWMA chart (A_t top-r absolute-residual mean or V_t log-sample-variance).
Returns the run length at which the chart signals.
"""
import numpy as np
from functions.drifting_functions import get_y_value
from functions.sampling import get_samples
from functions.monitoring import EWMA_UCL


# =============================================================================
# Drift-region setup
# =============================================================================
def drift_setting(drift_introduction, drift_ratio, drift_features, boundaries, rng):
    """
    Place the center of the drift region so that its volume equals
    `drift_ratio` * (domain volume). Returns (drift_position, drift_w).
    """
    if drift_introduction:
        drift_w = drift_ratio ** (1 / drift_features) / 2 \
                  * (boundaries[0][1] - boundaries[0][0])
        drift_position = np.array([
            rng.uniform(b[0] + drift_w, b[1] - drift_w)
            for b in boundaries[:drift_features]
        ])
        return drift_position, drift_w
    return None, 0


# =============================================================================
# EWMA chart setup / update
# =============================================================================
def monitoring_setting(monitoring_dic, lamb):
    """Instantiate EWMA chart objects based on `monitoring_dic` keys."""
    detection_dic = {key: False for key in monitoring_dic.keys()}
    chart_dic = {}

    if 'average' in monitoring_dic:
        chart_dic['average'] = EWMA_UCL(
            monitoring_dic['average']['target_a'],
            monitoring_dic['average']['UCL'],
            monitoring_dic['average']['r_largest'], lamb)

    if 'variance' in monitoring_dic:
        chart_dic['variance'] = EWMA_UCL(
            monitoring_dic['variance']['target_v'],
            monitoring_dic['variance']['UCL'], 0, lamb)

    return chart_dic, detection_dic


def monitoring_update(chart_dic, detection_dic, new_residual):
    """Advance each active EWMA chart by one step."""
    # A_t = (1/r) * sum_{j=1}^{r} |e|_{(j)} : top-r absolute-residual mean
    if not detection_dic.get('average', True):
        r = chart_dic['average'].r_largest
        val = np.partition(np.abs(new_residual), -r)[-r:].mean()
        chart_dic['average'].add_monitoring_val(val)
        detection_dic['average'] = chart_dic['average'].oc_decision()

    # V_t = log(s^2) : log sample variance of the batch residuals
    if not detection_dic.get('variance', True):
        chart_dic['variance'].add_monitoring_val(np.log(np.var(new_residual, ddof=1)))
        detection_dic['variance'] = chart_dic['variance'].oc_decision()

    return chart_dic, detection_dic


# =============================================================================
# Full experiment loop
# =============================================================================
def experiments_with_model(function, noise, boundaries, epsilon, budget, monitoring_dic,
                           prediction_model, x_available, y_available,
                           drift_introduction=0, drift_magnitudes=0, drift_ratio=0,
                           grid_res=20, lamb=0.2, n_init_sample=10000,
                           bandwidth=None, seed=None):
    """
    Run one experiment. Abrupt mean drift is introduced at 'drift_introduction`.
    The loop continues until the chart signals;
    the returned run length is the total number of steps taken.
    """
    rng = np.random.default_rng(int(seed))

    # --- Initialization ---
    run_length = 0
    x_samples_available = x_available
    y_samples_available = y_available
    y_predict_available = prediction_model.predict(x_samples_available)

    drift_features = len(boundaries)
    drift_position, drift_w = drift_setting(
        drift_introduction, drift_ratio, drift_features, boundaries, rng)

    chart_dic, detection_dic = monitoring_setting(monitoring_dic, lamb)
    last_zeroed_time = {}

    # --- running loop ---
    while True:
        run_length += 1

        # Keep only the most recent n_init_sample points (sliding window).
        x_samples_available = x_samples_available[-n_init_sample:]
        y_samples_available = y_samples_available[-n_init_sample:]
        y_predict_available = y_predict_available[-n_init_sample:]

        # Draw the new batch (exploration + exploitation).
        new_x_samples, last_zeroed_time = get_samples(
            budget, last_zeroed_time, run_length, boundaries, grid_res, epsilon,
            x_samples_available,
            weights=(y_samples_available - y_predict_available) ** 2,
            bandwidth=bandwidth, rng=rng)

        # Abrupt drift activation at drift_introduction.
        magnitudes_adj = np.where(run_length >= drift_introduction,
                                  drift_magnitudes, 0)

        # Sample the response and compute residuals.
        new_y_samples = get_y_value(
            new_x_samples, noise, function, magnitudes_adj,
            drift_position, 'partial', drift_w=drift_w, rng=rng)
        new_y_predict = prediction_model.predict(new_x_samples)
        new_residual = new_y_samples - new_y_predict

        # Update EWMA charts.
        chart_dic, detection_dic = monitoring_update(
            chart_dic, detection_dic, new_residual)

        # Stop when any active chart signals.
        if all(detection_dic.values()):
            break

        # Append new labels to the sliding pool.
        x_samples_available = np.vstack([x_samples_available, new_x_samples])
        y_samples_available = np.r_[y_samples_available, new_y_samples]
        y_predict_available = np.r_[y_predict_available, new_y_predict]

    return run_length, (chart_dic, detection_dic)
