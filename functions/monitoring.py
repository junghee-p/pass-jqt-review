"""
One-sided upper EWMA control chart used by PASS.

Applied to a scalar monitoring statistic: either the top-r absolute-residual
mean A_t or the log-sample-variance V_t.
"""
import numpy as np


class EWMA_UCL:
    """
    One-sided upper EWMA.

    Parameters
    ----------
    target : float
        In-control mean of the monitored statistic.
    UCL : float
        Upper control limit; a signal is issued when Z_t > UCL.
    r_largest : int
        Number of top-r absolute residuals averaged when computing A_t. Set to
        0 for the log-variance chart (not used).
    lamb : float
        EWMA smoothing weight (the paper uses 0.2).

    Update rule at each step t with new monitored value `val`:
        d   = val - target
        Z_t = lambda * max(0, d) + (1 - lambda) * Z_{t-1}     (one-sided upper)
    Signal if Z_t > UCL.
    """

    def __init__(self, target=0, UCL=None, r_largest=0, lamb=0.2):
        self.target = target
        self.UCL = UCL
        self.r_largest = int(r_largest)
        self.lamb = lamb
        self.current_idx = 0
        self.monitoring = [0]    # Z_0 = 0

    def add_monitoring_val(self, val):
        self.current_idx += 1
        d = val - self.target
        # max(0, .) turns this into the one-sided upper EWMA.
        self.monitoring.append(self.lamb * max(0, d)
                               + (1 - self.lamb) * self.monitoring[-1])

    def oc_decision(self):
        """Return True if the chart has signalled out of control."""
        return np.abs(self.monitoring[-1]) > self.UCL
