"""
Grid search demo: scan up/down/conf thresholds and print sorted results.

Usage:
    python examples/grid_search_demo.py --data ./data/your_etf_30m.csv --lookback 256 --mc_samples 6
"""
import argparse
import pandas as pd
import numpy as np

from backtest import (
    PredictorRunner,
    compute_step_return_samples,
    classify_mc_returns,
    grid_search,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="CSV with timestamps, open, high, low, close[, volume, amount]")
    parser.add_argument("--lookback", type=int, default=256)
    parser.add_argument("--mc_samples", type=int, default=6)
    parser.add_argument("--up_list", type=str, default="0.0005,0.0008,0.001")
    parser.add_argument("--down_list", type=str, default="0.0005,0.0008,0.001")
    parser.add_argument("--conf_list", type=str, default="0.55,0.6,0.65")
    parser.add_argument("--buy_fee", type=float, default=0.001)
    parser.add_argument("--sell_fee", type=float, default=0.0015)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--allow_short", action="store_true", help="enable short (ETF style)")
    parser.add_argument("--delay", type=int, default=0, help="execution delay bars (1 for T+1)")
    parser.add_argument("--bars_per_year", type=float, default=252 * 8)
    args = parser.parse_args()

    up_list = [float(x) for x in args.up_list.split(",")]
    down_list = [float(x) for x in args.down_list.split(",")]
    conf_list = [float(x) for x in args.conf_list.split(",")]

    df = pd.read_csv(args.data)
    df["timestamps"] = pd.to_datetime(df["timestamps"])

    runner = PredictorRunner(device=args.device, max_context=args.lookback)

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

    mc_ret_samples = np.stack(ret_samples_list, axis=0)
    close_series = pd.Series(close_exec)

    results = grid_search(
        mc_ret_samples=mc_ret_samples,
        close=close_series,
        up_thresh_list=up_list,
        down_thresh_list=down_list,
        conf_thresh_list=conf_list,
        allow_short=args.allow_short,
        delay=args.delay,
        buy_fee=args.buy_fee,
        sell_fee=args.sell_fee,
        bars_per_year=args.bars_per_year,
    )

    print("Top 10 grid search results (sorted by annual_return):")
    print(results.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
