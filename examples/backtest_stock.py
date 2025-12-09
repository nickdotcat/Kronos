"""
Example: A-share daily backtest (T+1, long-only) using Kronos MC forecasts.

Usage:
    python examples/backtest_stock.py --data ./data/your_stock_daily.csv --lookback 400 --mc_samples 8
"""
import argparse
import pandas as pd
import numpy as np

from backtest import (
    PredictorRunner,
    compute_step_return_samples,
    classify_mc_returns,
    run_backtest,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="CSV with columns timestamps, open, high, low, close[, volume, amount]")
    parser.add_argument("--lookback", type=int, default=400, help="context length (<= model max context)")
    parser.add_argument("--mc_samples", type=int, default=8, help="MC sample count")
    parser.add_argument("--up_thresh", type=float, default=0.002, help="long threshold")
    parser.add_argument("--conf_thresh", type=float, default=0.55, help="confidence threshold")
    parser.add_argument("--buy_fee", type=float, default=0.001, help="buy cost")
    parser.add_argument("--sell_fee", type=float, default=0.0015, help="sell cost")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    df["timestamps"] = pd.to_datetime(df["timestamps"])

    runner = PredictorRunner(device=args.device)

    ret_samples_list = []
    close_exec = []
    for idx in range(args.lookback, len(df) - 1):
        ctx = df.iloc[idx - args.lookback : idx]
        future_ts = df["timestamps"].iloc[idx : idx + 1]

        mc_prices, _ = runner.predict_mc(
            df=ctx[runner.predictor.price_cols + [runner.predictor.vol_col, runner.predictor.amt_vol]],
            x_timestamp=ctx["timestamps"],
            y_timestamp=future_ts,
            pred_len=1,
            mc_samples=args.mc_samples,
        )
        last_close = ctx["close"].iloc[-1]
        ret_samples = compute_step_return_samples(mc_prices, last_close, step=0)
        ret_samples_list.append(ret_samples)
        close_exec.append(df["close"].iloc[idx])

    mc_ret_samples = np.stack(ret_samples_list, axis=0)  # (n_steps, mc_samples)
    close_series = pd.Series(close_exec)

    signals, _, _ = classify_mc_returns(
        mc_ret_samples,
        up_thresh=args.up_thresh,
        down_thresh=0.0,
        conf_thresh=args.conf_thresh,
        allow_short=False,
    )

    bt = run_backtest(
        close=close_series,
        signals=signals,
        buy_fee=args.buy_fee,
        sell_fee=args.sell_fee,
        allow_short=False,
        delay=1,  # T+1 execution
        bars_per_year=252,
    )

    print("Backtest metrics:")
    for k, v in bt["metrics"].items():
        print(f"{k}: {v:.6f}")


if __name__ == "__main__":
    main()
