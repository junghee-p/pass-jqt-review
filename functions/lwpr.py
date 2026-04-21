"""
Locally weighted polynomial regression (LWPR) used in the UK electricity
market case study.

Given training features X_train and targets y_train, a prediction at a query
point x* is produced by:
  1) picking an adaptive bandwidth as a percentile of distances from x* to X_train;
  2) weighting training points by a tri-cube kernel on scaled distances;
  3) solving a weighted least-squares regression with a polynomial expansion;
  4) evaluating the fitted polynomial at x*.
"""
import torch
from sklearn.preprocessing import MinMaxScaler


# =============================================================================
# Basis and kernel
# =============================================================================
def polynomial_features(x, order=2):
    """Polynomial feature map for 2-D inputs up to the given total degree."""
    n = x.size(0)
    features = [torch.ones(n)]
    for i in range(1, order + 1):
        for j in range(i + 1):
            features.append((x[:, 0] ** (i - j)) * (x[:, 1] ** j))
    return torch.cat([f.unsqueeze(1) for f in features], dim=1)


def tri_cube(u):
    """Tri-cube weight function: (1 - u^3)^3 for u < 1, 0 otherwise."""
    return torch.where(u < 1, (1 - u ** 3) ** 3, torch.zeros_like(u))


def calculate_weights(data_points, fitting_point, bandwidth):
    """Tri-cube weights on scaled Euclidean distances."""
    if fitting_point.dim() == 1:
        fitting_point = fitting_point.unsqueeze(0)
    distances = torch.sqrt(torch.sum((data_points - fitting_point) ** 2, axis=1))
    return tri_cube(distances / bandwidth)


def adaptive_bandwidth(data_points, fitting_point, percentile=30):
    """Adaptive bandwidth: the `percentile`-th quantile of distances from `fitting_point`."""
    if fitting_point.dim() == 1:
        fitting_point = fitting_point.unsqueeze(0)
    distances = torch.sqrt(torch.sum((data_points - fitting_point) ** 2, axis=1))
    return torch.quantile(distances, percentile / 100.0)


# =============================================================================
# Weighted least squares
# =============================================================================
def weighted_least_squares(X, y, weights):
    """Solve min_beta sum_i w_i (y_i - x_i^T beta)^2 via the pseudoinverse."""
    weighted_X = X * weights.unsqueeze(1)
    XTWX = weighted_X.T @ X
    XTWy = weighted_X.T @ y
    beta = torch.linalg.pinv(XTWX) @ XTWy
    return beta.squeeze()


# =============================================================================
# Public API
# =============================================================================
def process_data(data, covariates, response):
    """Scale covariates to [-1, 1]; return (X_tensor, y_tensor, fitted scaler)."""
    scaler = MinMaxScaler(feature_range=(-1, 1))
    data[covariates] = scaler.fit_transform(data[covariates])
    features = torch.tensor(data[covariates].values, dtype=torch.float32)
    target = torch.tensor(data[response].values, dtype=torch.float32)
    return features, target, scaler


def predict_LWPR(features_train, target_train, new_features, percentile=30, order=2):
    """
    LWPR prediction.

    features_train : (N, d) tensor of training inputs.
    target_train   : (N,)   tensor of training targets.
    new_features   : (M, d) tensor of query points.
    percentile     : percentile used to set the adaptive bandwidth.
    order          : polynomial order.
    """
    X_poly_base = polynomial_features(features_train, order=order)

    predictions = []
    for i in range(new_features.size(0)):
        x_query = new_features[i]

        bandwidth = adaptive_bandwidth(features_train, x_query, percentile=percentile)
        weights = calculate_weights(features_train, x_query, bandwidth)
        beta = weighted_least_squares(X_poly_base, target_train, weights)

        X_poly_query = polynomial_features(x_query.unsqueeze(0), order=order)
        predictions.append(float(torch.matmul(X_poly_query, beta).item()))

    return predictions
