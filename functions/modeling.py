"""
Predictor class used by the pre-fit Branin example predictor.

This module exists so that `joblib.load('Branin_predictor.pkl')` can find the
class `SplineInteractionRegressor` at unpickling time. Only the subset of the
original implementation needed to serve `.predict(X)` on a loaded instance is
kept here.
"""
import numpy as np
from sklearn.preprocessing import PolynomialFeatures, SplineTransformer
from sklearn.linear_model import Ridge


class SplineInteractionRegressor:
    """
    Per-axis B-spline bases + pairwise interactions + Ridge regression.

    A fitted instance carries:
        _tr_list           : list of fitted SplineTransformer objects, one per axis.
        _poly              : fitted PolynomialFeatures (interaction_only=True).
        _poly_keep_idx_    : column indices retained after post-hoc filtering.
        lin                : fitted Ridge regressor.
        _n_features_out    : number of retained features (= len of _poly_keep_idx_).
    """

    def __init__(self, **kwargs):
        # Real fitting happens elsewhere; a loaded pickle overrides all attributes.
        self._tr_list = None
        self._poly = None
        self._poly_keep_idx_ = None
        self._n_features_out = None
        self.lin = None

    def _design(self, X):
        """Build the filtered design matrix Z (n, p)."""
        B_list = [tr.transform(X[:, [j]]) for j, tr in enumerate(self._tr_list)]
        Z_cat = np.hstack(B_list)
        M = self._poly.transform(Z_cat)
        return M[:, self._poly_keep_idx_]

    def predict(self, X):
        return self.lin.predict(self._design(X))
