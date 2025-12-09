import numpy as np
from typing import Tuple


def classify_mc_returns(
    ret_samples: np.ndarray,
    up_thresh: float,
    down_thresh: float,
    conf_thresh: float,
    allow_short: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate ternary signals from MC return samples.

    Args:
        ret_samples: shape (n_steps, n_samples)
        up_thresh: minimum expected return to trigger long
        down_thresh: minimum magnitude to trigger short
        conf_thresh: minimum probability mass supporting the direction
        allow_short: whether to emit -1 signals

    Returns:
        signals: array of shape (n_steps,), values in {-1, 0, 1}
        prob_up: P(ret > 0)
        prob_down: P(ret < 0)
    """
    mean_ret = ret_samples.mean(axis=1)
    prob_up = (ret_samples > 0).mean(axis=1)
    prob_down = (ret_samples < 0).mean(axis=1)

    signals = np.zeros_like(mean_ret)
    long_mask = (mean_ret >= up_thresh) & (prob_up >= conf_thresh)
    if allow_short:
        short_mask = (mean_ret <= -down_thresh) & (prob_down >= conf_thresh)
    else:
        short_mask = np.zeros_like(long_mask, dtype=bool)

    signals[long_mask] = 1
    signals[short_mask] = -1
    return signals, prob_up, prob_down
