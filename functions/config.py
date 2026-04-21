"""
Central configuration for PASS experiments.

Per-function settings (domains, noise, training size, KDE bandwidth) and common
experiment settings (budget, EWMA smoothing, drift introduction time).

Model hyperparameters and calibrated UCLs are NOT stored here; the simulation
script loads a pre-fit predictor from disk and uses a UCL constant provided
in the script itself.
"""
import numpy as np


# =============================================================================
# Supported test functions
# =============================================================================
FUNCTIONS = ['branin_2d', 'ishigami_3d', 'friedman_5d', 'linkletter_8d', 'welch_20d']


# =============================================================================
# Input boundaries [lower, upper] per dimension
# =============================================================================
BOUNDARIES = {
    'branin_2d':     [(-5, 10), (0, 15)],
    'ishigami_3d':   [(-np.pi, np.pi)] * 3,
    'friedman_5d':   [(0, 10)] * 5,
    'linkletter_8d': [(0, 1)] * 8,
    'welch_20d':     [(-0.5, 0.5)] * 20,
}


# =============================================================================
# Observation noise standard deviation
# =============================================================================
NOISE = {
    'branin_2d':     11.32,
    'ishigami_3d':   0.187,
    'friedman_5d':   1.0,
    'linkletter_8d': 0.05,
    'welch_20d':     0.5,
}


# =============================================================================
# Phase-I training sample size (size of the reference sample used to estimate
# sigma_hat and the in-control target of the monitoring statistic)
# =============================================================================
N_INIT_SAMPLE = {
    'branin_2d':     3500,
    'ishigami_3d':   5250,
    'friedman_5d':   20000,
    'linkletter_8d': 10000,
    'welch_20d':     20000,
}


# =============================================================================
# Grid resolution per axis for the exploration accept-reject rule
# =============================================================================
GRID_RES = {
    'branin_2d':     100,
    'ishigami_3d':   25,
    'friedman_5d':   10,
    'linkletter_8d': 5,
    'welch_20d':     5,
}


# =============================================================================
# KDE bandwidth for the exploitation sampler
# =============================================================================
KDE_BANDWIDTH = {
    'branin_2d':     0.2,
    'ishigami_3d':   0.12,
    'friedman_5d':   0.16,
    'linkletter_8d': 0.03,
    'welch_20d':     0.01,
}


# =============================================================================
# Monitoring strategies as (label, epsilon, chart_type).
# epsilon = share of the budget allocated to exploration.
# chart_type in {'variance', 'average'}.
# =============================================================================
STRATEGIES = [
    ('PASS',   0.5, 'variance'),     # PASS, balanced
    ('Random', 1.0, 'variance'),     # Pure random sampling
]


# =============================================================================
# Common experiment settings
# =============================================================================
BUDGET = 20                # Number of labels per time step
LAMBDA = 0.2               # EWMA smoothing parameter
DRIFT_INTRODUCTION = 30    # Time step at which drift is introduced
