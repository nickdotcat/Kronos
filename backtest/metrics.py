import numpy as np
import pandas as pd
from typing import Dict


def annualize_return(returns: np.ndarray, bars_per_year: float) -> float:
    if returns.size == 0:
        return 0.0
    cum_ret = np.prod(1 + returns) - 1
    periods = returns.size
    return (1 + cum_ret) ** (bars_per_year / periods) - 1 if periods > 0 else 0.0


def sharpe_ratio(returns: np.ndarray, bars_per_year: float, eps: float = 1e-9) -> float:
    if returns.size == 0:
        return 0.0
    mean = returns.mean()
    std = returns.std()
    if std < eps:
        return 0.0
    return (mean / std) * np.sqrt(bars_per_year)


def max_drawdown(equity: np.ndarray) -> float:
    if equity.size == 0:
        return 0.0
    running_max = np.maximum.accumulate(equity)
    drawdowns = 1 - equity / (running_max + 1e-12)
    return float(drawdowns.max())


def trade_stats(trade_returns: np.ndarray) -> Dict[str, float]:
    if trade_returns.size == 0:
        return {
            "avg_trade_ret_net": 0.0,
            "win_rate": 0.0,
        }
    wins = (trade_returns > 0).sum()
    win_rate = wins / trade_returns.size
    return {
        "avg_trade_ret_net": float(trade_returns.mean()),
        "win_rate": float(win_rate),
    }


def precision_metrics(
    positions: np.ndarray,
    per_bar_pnl: np.ndarray,
) -> Dict[str, float]:
    up_mask = positions > 0
    down_mask = positions < 0

    def _precision(mask):
        if mask.sum() == 0:
            return 0.0
        return float((per_bar_pnl[mask] > 0).mean())

    strict_wrong_up = per_bar_pnl[up_mask][per_bar_pnl[up_mask] < 0]
    strict_wrong_down = per_bar_pnl[down_mask][per_bar_pnl[down_mask] < 0]

    return {
        "precision_up_net": _precision(up_mask),
        "precision_down_net": _precision(down_mask),
        "strict_wrong_up_loss": float(strict_wrong_up.mean()) if strict_wrong_up.size else 0.0,
        "strict_wrong_down_loss": float(strict_wrong_down.mean()) if strict_wrong_down.size else 0.0,
    }
