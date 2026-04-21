"""
Example: run a single PASS (or Random) monitoring experiment on Branin and
report the detection delay.

This script is a minimal demonstration of the PASS workflow used for the
simulation study. The full grid (all test functions, drift magnitudes and
ratios, hundreds of replications, plus ARL0 calibration by bisection) is
produced by a SLURM array job that loops over this same core routine; that
orchestration is omitted here for clarity.

USAGE
-----
    python run_simulation_example.py

HOW TO SWITCH STRATEGIES
------------------------
Edit the parameters in the "Experiment settings" block below. Each variant is
characterized by:
    strategy   in {'PASS', 'Random'}
    epsilon    in [0, 1] (share of the budget allocated to exploration)
    monitoring in {'variance', 'average'}

The two configurations used in the paper are:
    PASS  : strategy='PASS',   epsilon=0.5, monitoring='variance'
    Random: strategy='Random', epsilon=1.0, monitoring='variance'
"""
import sys
sys.path.insert(0, '.')

from functions.config import (
    BOUNDARIES, NOISE, N_INIT_SAMPLE, GRID_RES, KDE_BANDWIDTH,
    BUDGET, LAMBDA, DRIFT_INTRODUCTION,
)
from functions.utils import model_initialization, monitoring_initialization
from functions.experiments import experiments_with_model


# =============================================================================
# Experiment settings (edit here to switch strategy / drift setting)
# =============================================================================
function         = 'branin_2d'          # name in functions.config.FUNCTIONS
predictor_path   = 'Branin_predictor.pkl'

strategy         = 'PASS'               # 'PASS' or 'Random'
epsilon          = 0.5                  # 1.0 = pure random sampling
monitoring_type  = 'variance'           # 'variance' or 'average'

drift_ratio      = 0.01                 # 1 % of the input domain is affected
drift_magnitude  = 2.0                  # Delta in units of sigma
seed             = 0


# =============================================================================
# Calibrated UCL (example value for Branin + PASS + variance chart)
# =============================================================================
# The UCL below was calibrated by bisection to target ARL0 = 200 on the
# Branin function with PASS (epsilon = 0.5) and the log-variance chart.
# For any other (function, strategy, epsilon, monitoring) cell, run an ARL0
# calibration loop: simulate in-control (drift_magnitudes = 0), bisect on the
# UCL, and stop once the empirical ARL0 over 200 replications equals 200
# within tolerance. See the NOTE at the bottom of this script for the sketch.
UCL = 0.316526


# =============================================================================
# 1. Load the pre-fit predictor and generate the reference sample
# =============================================================================
print(f"=== {function} | {strategy} eps={epsilon} mon={monitoring_type} ===")
print("Step 1: load predictor and draw reference sample ...")

noise      = NOISE[function]
boundaries = BOUNDARIES[function]
n_init     = N_INIT_SAMPLE[function]
grid_res   = GRID_RES[function]
bandwidth  = KDE_BANDWIDTH[function]

prediction_model, x_initial, y_initial = model_initialization(
    function, boundaries, n_init, noise, predictor_path, seed=seed)

# =============================================================================
# 2. Build the monitoring dictionary with the user-supplied UCL
# =============================================================================
print("Step 2: set up the EWMA monitoring chart ...")

monitoring_dic, prediction_model, sigma_hat = monitoring_initialization(
    monitoring_type, prediction_model, x_initial, y_initial,
    SLIDING_WINDOW_LEN, BUDGET, UCL=UCL, seed=seed)
print(f"  sigma_hat = {sigma_hat:.4f} | UCL = {UCL}")


# Truncate the reference sample to the sliding-window length used in Phase II.
# The first n_init points were used to fit `prediction_model`; from here on
# only the most recent SLIDING_WINDOW_LEN points seed the monitoring pool.
SLIDING_WINDOW_LEN = min(2000, n_init)
x_initial = x_initial[-SLIDING_WINDOW_LEN:]
y_initial = y_initial[-SLIDING_WINDOW_LEN:]


# =============================================================================
# 3. Run one Phase-II experiment
# =============================================================================
# Drift is introduced at step DRIFT_INTRODUCTION. The loop inside
# `experiments_with_model` stops as soon as the chart signals; the returned
# run length is the total number of steps taken.
print(f"Step 3: simulate Phase-II (drift at step {DRIFT_INTRODUCTION}) ...")

run_length, _ = experiments_with_model(
    function, noise, boundaries, epsilon, BUDGET, monitoring_dic,
    prediction_model, x_initial, y_initial,
    drift_introduction=DRIFT_INTRODUCTION,
    drift_magnitudes=drift_magnitude,
    drift_ratio=drift_ratio,
    grid_res=grid_res,
    n_init_sample=SLIDING_WINDOW_LEN,
    bandwidth=bandwidth,
    seed=seed,
    lamb=LAMBDA,
)

detection_delay = run_length - DRIFT_INTRODUCTION
print(f"\nRun length     : {run_length}")
print(f"Detection delay: {detection_delay} steps after drift onset")


# =============================================================================
# NOTE on ARL0 calibration
# =============================================================================
# To calibrate a fresh UCL for a new (function, strategy, epsilon, monitoring)
# cell, loop the in-control experiment (drift_magnitudes = 0) under candidate
# UCL values and use bisection to hit ARL0 = 200 (+/- tolerance):
#
#     def eval_arl0(ucl_value):
#         md, pm, sig = monitoring_initialization(
#             monitoring_type, pm, x_initial, y_initial,
#             n_init, BUDGET, UCL=ucl_value)
#         run_lengths = []
#         for s in range(200):
#             rl, _ = experiments_with_model(
#                 function, noise, boundaries, epsilon, BUDGET, md,
#                 pm, x_initial, y_initial,
#                 drift_introduction=0, drift_magnitudes=0, drift_ratio=0,
#                 grid_res=grid_res, n_init_sample=min(2000, n_init),
#                 bandwidth=bandwidth, seed=s)
#             run_lengths.append(rl)
#         return float(np.mean(run_lengths))
#
# Bisect on `ucl_value` until eval_arl0(ucl_value) matches the target ARL0.
