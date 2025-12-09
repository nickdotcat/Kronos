import itertools
from typing import List, Dict, Any

import numpy as np
import pandas as pd

from .signal import classify_mc_returns
from .engine import run_backtest


def grid_search(
    mc_ret_samples: np.ndarray,
    close: pd.Series,
    up_thresh_list: List[float],
    down_thresh_list: List[float],
    conf_thresh_list: List[float],
    allow_short: bool,
    delay: int,
    buy_fee: float,
    sell_fee: float,
    bars_per_year: float,
) -> pd.DataFrame:
    """
    Evaluate parameter grid and return metrics per combination.

    Args:
        mc_ret_samples: shape (n_steps, n_samples) Monte Carlo return samples per bar
        close: close price series aligned to mc_ret_samples (length n_steps)
    """
    results: List[Dict[str, Any]] = []
    for up, down, conf in itertools.product(up_thresh_list, down_thresh_list, conf_thresh_list):
        signals, prob_up, prob_down = classify_mc_returns(
            mc_ret_samples, up, down, conf, allow_short=allow_short
        )
        bt = run_backtest(
            close=close,
            signals=signals,
            buy_fee=buy_fee,
            sell_fee=sell_fee,
            allow_short=allow_short,
            delay=delay,
            bars_per_year=bars_per_year,
        )
        row = {
            "up_thresh": up,
            "down_thresh": down,
            "conf_thresh": conf,
        }
        row.update(bt["metrics"])
        results.append(row)

    df = pd.DataFrame(results)
    df = df.sort_values(by="annual_return", ascending=False).reset_index(drop=True)
    return df
