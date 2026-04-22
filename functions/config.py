"""
Central configuration for the Branin simulation example.

Per-function settings (domain, noise, training size, KDE bandwidth) and common
experiment settings (budget, EWMA smoothing, drift introduction time).
"""
import numpy as np


# =============================================================================
# Supported test functions
# =============================================================================
FUNCTIONS = ['branin_2d']


# =============================================================================
# Input boundaries [lower, upper] per dimension
# =============================================================================
BOUNDARIES = {
    'branin_2d': [(-5, 10), (0, 15)],
}


# =============================================================================
# Observation noise standard deviation
# =============================================================================
NOISE = {
    'branin_2d': 11.32,
}


# =============================================================================
# Training sample size (size of the reference sample used to estimate sigma_hat
# and the in-control target of the monitoring statistic)
# =============================================================================
N_INIT_SAMPLE = {
    'branin_2d': 3500,
}


# =============================================================================
# Grid resolution per axis for the exploration accept-reject rule
# =============================================================================
GRID_RES = {
    'branin_2d': 100,
}


# =============================================================================
# KDE bandwidth for the exploitation sampler
# =============================================================================
KDE_BANDWIDTH = {
    'branin_2d': 0.2,
}


# =============================================================================
# Monitoring strategies as (label, epsilon, chart_type).
# epsilon   = share of the budget allocated to exploration.
# chart_type in {'variance', 'average'}.
# =============================================================================
STRATEGIES = [
    ('PASS',   0.5, 'variance'),
]


# =============================================================================
# Common experiment settings
# =============================================================================
BUDGET = 20                # Number of labels per time step
LAMBDA = 0.2               # EWMA smoothing parameter
DRIFT_INTRODUCTION = 30    # Time step at which drift is introduced
