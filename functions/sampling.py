"""
Adaptive sampling routines: exploitation via residual-weighted KDE and
exploration via accept-reject on a sparse last-visit map.

Entry point is `get_samples`, which splits the budget between the two routines
according to the exploration ratio `epsilon`.
"""
import numpy as np
from sklearn.neighbors import KernelDensity


# =============================================================================
# Exploitation: residual-weighted KDE
# =============================================================================
def adaptive_sampling(x_available, residuals, boundaries, n_exploitation, bandwidth):
    """
    Draw `n_exploitation` points from a KDE fitted on `x_available` with weights
    proportional to `residuals` (squared residuals in practice). Samples that
    fall outside the boundaries are reflected back inside along that axis.
    """
    # 'scott' is a placeholder; we overwrite bandwidth_ explicitly below.
    kde = KernelDensity(bandwidth='scott').fit(x_available, sample_weight=residuals)
    kde.bandwidth_ = bandwidth

    new_kde_samples = kde.sample(n_exploitation, random_state=0)

    # Reflect out-of-domain samples.
    for d in range(len(boundaries)):
        lo, hi = boundaries[d]
        below = new_kde_samples[:, d] < lo
        above = new_kde_samples[:, d] > hi
        new_kde_samples[below, d] = 2 * lo - new_kde_samples[below, d]
        new_kde_samples[above, d] = 2 * hi - new_kde_samples[above, d]

    return new_kde_samples


# =============================================================================
# Exploration: accept-reject on a sparse last-visit map
# =============================================================================
def random_sampling(last_zeroed_time, run_length, boundaries, grid_res,
                    cell_width, n_exploration, rng):
    """
    Draw `n_exploration` points via accept-reject.

    For each candidate cell c, the acceptance probability is
        p_c = clip((t - tau_c) / min(B^d, t), 0, 1),
    where tau_c is the last visit time (0 if never visited).
    """
    n_features = len(boundaries)

    def cell_value(cell_index):
        t0 = last_zeroed_time.get(cell_index, 0)
        return run_length - t0

    new_rnd_samples = []
    while len(new_rnd_samples) < n_exploration:
        # Propose a cell uniformly among all |G| = grid_res^d cells.
        cell_idx = tuple(rng.integers(0, grid_res) for _ in range(n_features))
        val = cell_value(cell_idx)
        if rng.random() < np.clip(val / min(grid_res ** n_features, run_length), 0, 1):
            # Draw a uniform point inside the accepted cell.
            sample = np.array([
                rng.uniform(boundaries[d][0] + cell_idx[d] * cell_width[d],
                            boundaries[d][0] + (cell_idx[d] + 1) * cell_width[d])
                for d in range(n_features)
            ])
            last_zeroed_time = _update_zeroed_time(
                [sample], last_zeroed_time, run_length + 1,
                boundaries, grid_res, cell_width)
            new_rnd_samples.append(sample)

    return np.array(new_rnd_samples), last_zeroed_time


def _update_zeroed_time(x_samples, last_zeroed_time, zeroed_time,
                        boundaries, grid_res, cell_width):
    """Set tau_c := zeroed_time for the cell containing each sampled point."""
    for x in x_samples:
        idx = tuple(np.clip(int((x[d] - boundaries[d][0]) // cell_width[d]),
                            0, grid_res - 1)
                    for d in range(len(boundaries)))
        last_zeroed_time[idx] = zeroed_time
    return last_zeroed_time


# =============================================================================
# Combined sampler (exploration + exploitation)
# =============================================================================
def get_samples(budget, last_zeroed_time, run_length, boundaries, grid_res, epsilon,
                x_samples_available, weights=None, bandwidth=None, rng=None):
    """
    Draw `budget` points total, splitting into
        m_e = floor(budget * epsilon)     for exploration
        m_x = budget - m_e                 for exploitation.

    Special cases:
        epsilon == 1 : pure uniform random sampling.
        m_x == 0     : exploration only.
        m_e == 0     : exploitation only.
    """
    if rng is None:
        rng = np.random.default_rng(rng)

    # Purely uniform random sampling
    if epsilon == 1:
        return (np.column_stack([rng.uniform(b[0], b[1], budget) for b in boundaries]),
                last_zeroed_time)

    n_exploration = int(budget * epsilon)
    n_exploitation = budget - n_exploration
    cell_width = [(b[1] - b[0]) / grid_res for b in boundaries]

    if n_exploitation and n_exploration:
        # Exploitation first, then exploration (which also updates the visit map).
        new_kde_samples = adaptive_sampling(x_samples_available, weights, boundaries,
                                            n_exploitation, bandwidth)
        last_zeroed_time = _update_zeroed_time(
            new_kde_samples, last_zeroed_time, run_length + 1,
            boundaries, grid_res, cell_width)
        new_rnd_samples, last_zeroed_time = random_sampling(
            last_zeroed_time, run_length, boundaries, grid_res, cell_width,
            n_exploration, rng)
        new_x_samples = np.vstack([new_kde_samples, new_rnd_samples])
    elif n_exploitation:
        new_x_samples = adaptive_sampling(x_samples_available, weights, boundaries,
                                          n_exploitation, bandwidth)
    else:
        new_x_samples, last_zeroed_time = random_sampling(
            last_zeroed_time, run_length, boundaries, grid_res, cell_width,
            n_exploration, rng)

    return new_x_samples, last_zeroed_time
