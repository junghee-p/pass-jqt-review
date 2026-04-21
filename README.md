# PASS: Probabilistic Adaptive Sampling Strategy

Reference implementation for the paper
**"An Adaptive Sampling Framework for Detecting Localized Concept Drift under Label Scarcity"**
(submitted to *Journal of Quality Technology*, manuscript ID UJQT-2025-0178).

The repository provides a minimal, self-contained codebase that reproduces one
replication of each of the two evaluation tracks:

- a simulation experiment on the Branin benchmark function (Section 3);
- a case study on the UK electricity market (Section 4).

Both tracks share the core modules in `functions/`.


## Repository layout

```
.
├── functions/
│   ├── __init__.py
│   ├── config.py                 # Per-function settings and common constants
│   ├── drifting_functions.py     # Benchmark test functions + abrupt mean drift
│   ├── sampling.py               # PASS exploration and exploitation
│   ├── monitoring.py             # One-sided EWMA control chart
│   ├── experiments.py            # Phase-II experiment loop
│   ├── utils.py                  # Model loading, reference sample
│   ├── modeling.py               # Predictor class for the pre-fit Branin pkl
│   └── lwpr.py                   # LWPR used in the case study
├── data/
│   ├── electricity_train.csv     # Raw 2020 training data
│   └── electricity_test.csv      # Raw 2022 test data
├── Branin_predictor.pkl          # Pre-fit predictor used in the simulation example
├── run_simulation_example.py     # Single simulation experiment on Branin
├── run_case_study_example.py     # UK electricity case study (PASS)
├── requirements.txt
└── README.md
```


## Installation

```bash
pip install -r requirements.txt
```

Tested with Python 3.10+.


## Quick start

### 1. Simulation experiment

```bash
python run_simulation_example.py
```

Runs one replication of PASS on the Branin 2D function with drift magnitude
Δ = 2.0 and drift ratio π_d = 0.01, and prints the run length and detection
delay.

Expected output (seed = 0):

```
=== branin_2d | PASS eps=0.5 mon=variance ===
Step 1: load predictor and draw reference sample ...
Step 2: set up the EWMA monitoring chart ...
  sigma_hat = 11.3769 | UCL = 0.316526
Step 3: simulate Phase-II (drift at step 30) ...

Run length     : 52
Detection delay: 22 steps after drift onset
```

The UCL in the example is a pre-calibrated value (targeting ARL0 = 200) for
the Branin + PASS + variance cell; see the NOTE at the bottom of the script
for how to calibrate UCLs for other cells. The reference sample is truncated
to 2,000 points after the predictor is loaded, matching the sliding-window
length used during Phase II.


### 2. Case study

```bash
python run_case_study_example.py
```

Reproduces the PASS scenario from Section 4 on the UK electricity market
data. LWPR predictions are computed on the fly (no pre-computed prediction
files), so the script takes on the order of a minute to run.

Expected output (seed = 0, sliding-window length 1000):

```
In-control target = 0.0323 | UCL = 0.105
(UCL determined experimentally on 2020 baseline data.)

=== Detection summary ===
PASS first alarms: ['2022-03-09', '2022-08-16', '2022-09-12']
```


## Predictor file

`Branin_predictor.pkl` stores the pre-fit predictor used in the simulation
example. The class used in the shipped pkl is defined in
`functions/modeling.py` so that `joblib.load` can resolve it at unpickle
time. Any object exposing `.predict(X)` can be used in its place: save it
with `joblib.dump(obj, 'Branin_predictor.pkl')` and the simulation script
will pick it up via `joblib.load`.


## Data

The case study uses half-hourly day-ahead electricity prices and solar
generation shares for the UK market.

`data/electricity_{train,test}.csv` contain the raw 2020 (train) and 2022
(test) observations with columns `Date`, `forecasted_solar_penetration`,
`Hour`, and `Day Ahead APX Price`. Day-ahead prices come from APX via
Elexon, and solar-generation statistics come from the UK National Energy
System Operator (NESO).


## Scope and reproducibility notes

- This repository is a reference implementation for review. It reproduces
  one replication of each experiment rather than the full grid used in the
  paper. The full grid (all test functions, drift magnitudes and ratios,
  hundreds of replications, plus the ARL0 calibration loop) is produced by
  a SLURM array job that loops over `experiments_with_model` with the
  strategies defined in `functions.config.STRATEGIES`.
- Phase-I initial samples are drawn uniformly at random for every test
  function (no grid overlay), matching the behavior required in higher
  dimensions.
