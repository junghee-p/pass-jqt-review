"""
UK electricity market case study.

Reproduces the PASS scenario from Section 4 of the paper:

    PASS with budget = 4 SPs/day, epsilon = 2/8, and top-r = 2

The monitoring statistic is the top-r absolute-residual mean

    A_t = (1/r) * sum_{j=1}^{r} |e_t|_{(j)},

where |e_t|_{(1)} >= ... >= |e_t|_{(r)} are the r largest absolute residuals at
time t. The one-sided EWMA control chart signals when Z_t exceeds UCL.

Predictions are computed on the fly with LWPR (locally weighted polynomial
regression); there are no pre-computed prediction files involved. Because
LWPR does a local fit at every query point, the script takes on the order of
a minute to run on the test window used in the paper.

USAGE
-----
    python run_case_study_example.py
"""
import sys
sys.path.insert(0, '.')

import numpy as np
import pandas as pd
import torch
from sklearn.neighbors import KernelDensity
from tqdm import tqdm

from functions.lwpr import process_data, predict_LWPR
from functions.monitoring import EWMA_UCL


# =============================================================================
# Settings
# =============================================================================
COVARIATES      = ['forecasted_solar_penetration', 'Hour']
RESPONSE_COL    = 'Day Ahead APX Price'
N_SP_PER_DAY    = 48

SEED            = 0

BUDGET          = 4                   # SPs sampled per day under budget constraint
EPSILON         = 1 / 4               # PASS exploration ratio
                                      # -> 1 exploration + 3 exploitation samples per day

R_LARGEST       = BUDGET // 2         # = 2
LAMBDA          = 0.2                 # EWMA smoothing weight
UCL             = 0.105               # UCL determined experimentally on 2020 data
                                      # (see Section 4 of the paper)

HISTORY_LEN     = 1000                # sliding-window cap on the PASS history


# =============================================================================
# Helpers
# =============================================================================
def top_r_abs_mean(residuals, r):
    """Paper formula A_t = (1/r) * sum_j |e_t|_{(j)}."""
    r = min(r, len(residuals))
    idx = np.argsort(np.abs(residuals))[::-1][:r]
    return float(np.mean(np.abs(residuals[idx])))


def first_alarm_of_each_run(signal_steps, day_record):
    """Keep the first step of each contiguous alarm run; return the days."""
    if not signal_steps:
        return []
    run_starts = [signal_steps[0]]
    prev = signal_steps[0]
    for s in signal_steps[1:]:
        if s - 1 != prev:
            run_starts.append(s)
        prev = s
    return [day_record[s - 1] for s in run_starts]


def prepare_frame(raw_path):
    """Load raw CSV, add `day` and `Minute_index` columns, drop incomplete days."""
    df = pd.read_csv(raw_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df['day'] = df['Date'].dt.date.astype(str)
    df['Minute_index'] = (df['Date'].dt.hour * 60 + df['Date'].dt.minute) // 30
    cnt = df.groupby('day').size()
    full_days = cnt[cnt == N_SP_PER_DAY].index
    return df[df['day'].isin(full_days)].reset_index(drop=True)


# -- PASS sampling on the 48-SP grid ------------------------------------------
def pass_exploration(last_zeroed_time, run_length, n_exploration, n_bins=N_SP_PER_DAY):
    """Accept-reject exploration on SP indices {0, ..., 47}."""
    def cell_value(idx):
        return run_length - last_zeroed_time.get(idx, 0)

    selected = []
    while len(selected) < n_exploration:
        c = np.random.randint(0, n_bins)
        if np.random.random() < np.clip(cell_value(c) / min(n_bins, run_length), 0, 1):
            selected.append(c)
            last_zeroed_time[c] = run_length + 1
    return selected, last_zeroed_time


def pass_exploitation(last_zeroed_time, run_length, x_history, residuals_sq,
                      n_exploitation, n_bins=N_SP_PER_DAY):
    """Residual-weighted KDE exploitation on SP indices, with reflection at boundaries."""
    kde = KernelDensity(bandwidth='scott').fit(x_history, sample_weight=residuals_sq)
    selected = []
    while len(selected) < n_exploitation:
        raw = float(kde.sample().ravel()[0])
        if raw < 0:
            raw = -raw
        if raw > n_bins:
            raw = 2 * n_bins - raw
        idx = max(0, min(int(np.floor(raw)), n_bins - 1))
        if idx not in selected:
            selected.append(idx)
            last_zeroed_time[idx] = run_length + 1
    return selected, last_zeroed_time


# =============================================================================
# 1. Load raw data
# =============================================================================
print("Loading data ...")
df_train_raw = prepare_frame("data/electricity_train.csv")
df_test_raw  = prepare_frame("data/electricity_test.csv")

n_days_test = len(df_test_raw) // N_SP_PER_DAY
print(f"  Training: {len(df_train_raw)} rows ({len(df_train_raw)//N_SP_PER_DAY} days)")
print(f"  Test:     {len(df_test_raw)} rows ({n_days_test} days)")


# =============================================================================
# 2. Fit the LWPR feature scaler on the training frame
# =============================================================================
# `process_data` scales the covariates to [-1, 1] in-place and returns torch
# tensors. The same scaler is reused for the test features.
X_train, y_train, scaler = process_data(df_train_raw.copy(), COVARIATES, RESPONSE_COL)
X_test = torch.tensor(scaler.transform(df_test_raw[COVARIATES]),
                      dtype=torch.float32)


# =============================================================================
# 3. LWPR predictions for the initial sliding-window history
# =============================================================================
# Only the most recent HISTORY_LEN training points seed the monitoring pool,
# so we only need LWPR predictions for those rows.
print(f"\nFitting LWPR on the last {HISTORY_LEN} training points ...")
train_tail = df_train_raw.iloc[-HISTORY_LEN:].reset_index(drop=True)
X_hist     = X_train[-HISTORY_LEN:]
y_hist     = y_train[-HISTORY_LEN:]
y_hat_hist = np.array(predict_LWPR(X_train, y_train, X_hist,
                                   percentile=30, order=2))


# =============================================================================
# 4. PASS monitoring loop
# =============================================================================
print("\n--- PASS (budget = 4 SPs/day) ---")
np.random.seed(SEED)

n_exploration  = int(BUDGET * EPSILON)     # 1
n_exploitation = BUDGET - n_exploration    # 3

# In-control target: mean of absolute training residuals, computed directly
# on the sliding-window initial history (no bootstrap required).
target = float(np.mean(np.abs(y_hist.numpy() - y_hat_hist)))
chart  = EWMA_UCL(target=target, UCL=UCL, r_largest=R_LARGEST, lamb=LAMBDA)
print(f"In-control target = {target:.4f} | UCL = {UCL}")
print("(UCL determined experimentally on 2020 baseline data.)")

# Sliding-window history: SP-index features, observed price, LWPR prediction.
x_history     = train_tail['Minute_index'].values.reshape(-1, 1)
y_history     = y_hist.numpy()
y_hat_history = y_hat_hist.copy()

pass_signal_steps, day_record = [], []
last_zeroed_time = {}

for t in tqdm(range(1, n_days_test + 1), desc='PASS'):
    day_record.append(df_test_raw.loc[(t - 1) * N_SP_PER_DAY, 'day'])

    # Sliding-window cap.
    x_history     = x_history[-HISTORY_LEN:]
    y_history     = y_history[-HISTORY_LEN:]
    y_hat_history = y_hat_history[-HISTORY_LEN:]

    # PASS sampling on today's 48 SPs.
    sp_kde, last_zeroed_time = pass_exploitation(
        last_zeroed_time, t, x_history,
        (y_history - y_hat_history) ** 2, n_exploitation)
    sp_rnd, last_zeroed_time = pass_exploration(
        last_zeroed_time, t, n_exploration)
    sp_selected = sp_kde + sp_rnd

    # LWPR predictions for the four sampled points of the day.
    row_ids   = (t - 1) * N_SP_PER_DAY + np.array(sp_selected)
    X_query   = X_test[row_ids]
    y_query   = df_test_raw.loc[row_ids, RESPONSE_COL].values
    y_hat_q   = np.array(predict_LWPR(X_train, y_train, X_query,
                                      percentile=30, order=2))
    residuals = y_query - y_hat_q

    chart.add_monitoring_val(top_r_abs_mean(residuals, R_LARGEST))
    if chart.oc_decision():
        pass_signal_steps.append(t)

    # Append today's four points to the sliding history.
    x_history     = np.vstack([x_history, np.array(sp_selected).reshape(-1, 1)])
    y_history     = np.r_[y_history, y_query]
    y_hat_history = np.r_[y_hat_history, y_hat_q]

pass_alarm_days = first_alarm_of_each_run(pass_signal_steps, day_record)

print(f"\n=== Detection summary ===")
print(f"PASS first alarms: {pass_alarm_days or '(no detection)'}")
