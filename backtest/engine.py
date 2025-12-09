import numpy as np
import pandas as pd
from typing import Dict, Any

from .metrics import (
    annualize_return,
    sharpe_ratio,
    max_drawdown,
    trade_stats,
    precision_metrics,
    profit_factor,
    coverage,
    confidence_buckets,
)


def _compute_trades(position: np.ndarray) -> np.ndarray:
    """Return trade ids for grouping consecutive same-side positions."""
    trade_id = np.zeros_like(position, dtype=int)
    current = 0
    for i in range(1, position.size):
        if position[i] != position[i - 1]:
            current += 1
        trade_id[i] = current
    return trade_id


def run_backtest(
    close: pd.Series,
    signals: np.ndarray,
    buy_fee: float = 0.001,
    sell_fee: float = 0.0015,
    allow_short: bool = False,
    delay: int = 0,
    bars_per_year: float = 252,
    prob_up: np.ndarray = None,
    prob_down: np.ndarray = None,
) -> Dict[str, Any]:
    """
    Simple vectorized backtest.

    Args:
        close: price series
        signals: desired positions {-1,0,1} aligned with close index
        buy_fee/sell_fee: proportional costs applied on position change
        allow_short: whether -1 is permitted
        delay: bars to delay signal execution (e.g., 1 for T+1)
        bars_per_year: annualization factor
    """
    close = close.reset_index(drop=True)
    signals = np.asarray(signals, dtype=float)
    if signals.shape[0] != close.shape[0]:
        raise ValueError("signals and close must have same length")

    # apply execution delay
    exec_signal = np.roll(signals, delay)
    exec_signal[:delay] = 0
    if not allow_short:
        exec_signal = np.clip(exec_signal, 0, 1)

    ret = close.pct_change().fillna(0).to_numpy()
    position = np.zeros_like(exec_signal)
    position[0] = 0
    for i in range(1, position.size):
        position[i] = exec_signal[i - 1]

    pnl = position * ret

    # trading costs when position changes
    position_change = np.diff(position, prepend=0)
    costs = np.zeros_like(position)
    enter_long = position_change > 0
    exit_long = position_change < 0

    costs[enter_long] = buy_fee
    costs[exit_long] = sell_fee

    # short side uses same fees
    pnl -= costs

    equity = np.cumprod(1 + pnl)

    # trade-level returns
    trade_id = _compute_trades(position)
    trade_df = pd.DataFrame({"trade": trade_id, "pnl": pnl, "pos": position})
    trade_returns = (
        trade_df.groupby("trade")
        .apply(lambda x: np.prod(1 + x["pnl"].to_numpy()) - 1)
        .to_numpy()
    )
    trade_returns = trade_returns[trade_returns != 0]  # drop idle trades

    metrics = {
        "annual_return": annualize_return(pnl, bars_per_year),
        "sharpe": sharpe_ratio(pnl, bars_per_year),
        "max_drawdown": max_drawdown(equity),
    }
    metrics.update(trade_stats(trade_returns))
    metrics.update(precision_metrics(position, pnl))
    metrics["profit_factor"] = profit_factor(pnl)
    metrics["coverage"] = coverage(signals)

    if prob_up is not None and prob_down is not None:
        # use max of directional prob as confidence
        conf = np.maximum(prob_up, prob_down)
        metrics.update(confidence_buckets(conf, pnl))

    return {
        "equity_curve": equity,
        "per_bar_pnl": pnl,
        "position": position,
        "metrics": metrics,
    }
