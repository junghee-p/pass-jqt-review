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
│   ├── experiments.py            # Experiment loop
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

The UCL in the example is a pre-calibrated value (targeting ARL0 = 200) for
the Branin + PASS + variance cell; see the NOTE at the bottom of the script
for how to calibrate UCLs for other cells.

The simulation example uses the Branin function; the paper reports the full
grid of test functions (Ishigami, Friedman, Linkletter, Welch), all of which
use the same code structure with different hyperparameters and predictor files.


### 2. Case study

```bash
python run_case_study_example.py
```

Reproduces the PASS scenario from Section 4 on the UK electricity market data.


## Predictor file

`Branin_predictor.pkl` stores the pre-fit predictor used in the simulation
example.


## Data

The case study uses half-hourly day-ahead electricity prices and solar
generation shares for the UK market.

`data/electricity_{train,test}.csv` contain the raw 2020 (train) and 2022
(test) observations with columns `Date`, `forecasted_solar_penetration`,
`Hour`, and `Day Ahead APX Price`.
