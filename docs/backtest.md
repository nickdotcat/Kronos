# Kronos 回测与网格搜索技术文档

## 背景与目标
- **A 股日线（T+1，长多）**：基于 Kronos 对下一交易日 MC 预测，阈值+置信度生成涨/平/跌信号，延迟一根执行，计成本/印花税。  
- **ETF 30 分钟（T+0，多空）**：信号即时，多空皆可，计成本。  
- **网格搜索**：对 up/down/conf 阈值组合遍历，输出回测指标并选优。

流程：MC 抽样预测 → 收益样本 → 三分类信号 → T+1/T+0 回测 → 指标 → 网格搜索。

## 模块概览
- `backtest/predictor_runner.py`：封装 `KronosPredictor` 多次抽样生成 MC 价格 `(mc_samples, pred_len, feat)`。  
- `backtest/signal.py`：收益样本 → 三分类信号（阈值+置信度，支持/禁用做空）。  
- `backtest/engine.py`：回测引擎，支持执行延迟、多空、买卖成本；用 close pct_change 估算收益，输出资金曲线与指标。  
- `backtest/metrics.py`：年化收益、夏普、最大回撤、平均单笔收益、胜率、precision_up/down、严格错误损失/次数、profit factor、coverage、置信度分桶均值。  
- `backtest/grid_search.py`：遍历 up/down/conf 组合，生成信号→回测→汇总指标 DataFrame。  
- 示例：`examples/backtest_stock.py`（日线 T+1 长多）、`examples/backtest_etf.py`（30m T+0 多空）、`examples/grid_search_demo.py`（阈值扫描演示）。

## 数据与依赖
- 依赖见 `requirements.txt`；需加载 `NeoQuasar/Kronos-small` 与 `NeoQuasar/Kronos-Tokenizer-base`（离线请预缓存）。  
- 数据列：`timestamps, open, high, low, close`；可选 `volume, amount`；时间戳会转为 pandas datetime。

## 关键接口（类/方法）
- `PredictorRunner(model_id="NeoQuasar/Kronos-small", tokenizer_id="NeoQuasar/Kronos-Tokenizer-base", device="cpu", max_context=512, T=1.0, top_p=0.9, top_k=0)`  
  - 参数含义：  
    - `model_id/tokenizer_id`：HF 模型与分词器标识；可换成本地路径。  
    - `device`：如 `cpu`/`cuda:0`/`mps:0`。  
    - `max_context`：最长上下文窗口。  
    - `T/top_p/top_k`：采样温度、核采样、Top-K 采样控制生成随机性。  
  - `predict_mc(df, x_timestamp, y_timestamp, pred_len, mc_samples)` → `(samples, mean_path)`；`mc_samples` 为抽样次数，`pred_len` 为预测步数（示例使用 1）。
- `compute_step_return_samples(mc_samples, last_close, step=0)`：用预测 close 与 `last_close` 计算第 `step` 步收益样本 `(mc_samples,)`。
- `classify_mc_returns(ret_samples, up_thresh, down_thresh, conf_thresh, allow_short, use_thresh_prob=True)`：  
  - `ret_samples` 形状 `(n_steps, mc_samples)`；  
  - `up_thresh/down_thresh`：期望收益阈值（正/负）；  
  - `conf_thresh`：方向概率阈值；  
  - `allow_short`：是否允许输出 -1 信号；  
  - `use_thresh_prob`：置信度为超阈值概率（默认）或简单正/负概率。  
  - 返回 `(signals, prob_up, prob_down)`，`signals` 形状 `(n_steps,)`，取值 {-1,0,1}。
- `run_backtest(close, signals, buy_fee, sell_fee, allow_short, delay, bars_per_year, prob_up=None, prob_down=None)`：  
  - `close`：与 `signals` 对齐的收盘价序列；  
  - `buy_fee/sell_fee`：买卖比例成本（如 0.001=0.1%）；  
  - `allow_short`：允许空头；  
  - `delay`：信号执行延迟（1 表示 T+1）；  
  - `bars_per_year`：年化因子（日线 252，30m 约 252*8）。  
  - `prob_up/prob_down`：可选，传入以输出置信度分桶指标。  
  - 返回 `{equity_curve, per_bar_pnl, position, metrics}`，metrics 额外包含 profit_factor、coverage、严格错误次数、置信度分桶均值。
- `grid_search(mc_ret_samples, close, up_thresh_list, down_thresh_list, conf_thresh_list, allow_short, delay, buy_fee, sell_fee, bars_per_year)`：对阈值组合遍历，返回指标 DataFrame（按年化收益降序）。

## 通用步骤
1. 滑窗：选 `lookback`（≤ max_context），构造历史窗口 `ctx` 与下一根时间戳 `future_ts`，记录当前 close。  
2. MC 预测：`mc_prices, _ = PredictorRunner.predict_mc(ctx_df, ctx_ts, future_ts, pred_len=1, mc_samples=N)`。  
3. 收益样本：`ret_samples = compute_step_return_samples(mc_prices, last_close)`，收集成 `(n_steps, mc_samples)`。  
4. 信号：`signals, prob_up, prob_down = classify_mc_returns(mc_ret_samples, up_thresh, down_thresh, conf_thresh, allow_short)`；A 股 `allow_short=False`，ETF `allow_short=True`。  
5. 回测：`run_backtest(close_series, signals, buy_fee, sell_fee, allow_short, delay, bars_per_year)`；A 股 `delay=1, bars_per_year=252`；ETF 30m `delay=0, bars_per_year≈252*8`。  
6. 网格搜索（可选）：调用 `grid_search(...)` 传阈值列表，对比指标。

## 网格搜索示例
```bash
python examples/grid_search_demo.py \
  --data ./data/your_etf_30m.csv \
  --lookback 256 \
  --mc_samples 6 \
  --up_list 0.0005,0.0008,0.001 \
  --down_list 0.0005,0.0008,0.001 \
  --conf_list 0.55,0.6,0.65 \
  --allow_short \
  --device cuda:0
```
- 支持传 `delay`（T+1 则设 1）、`bars_per_year`、`buy_fee/sell_fee`。输出按年化收益排序的前 10 组。

## 示例与参数
### 日线 A 股（T+1，长多）
```bash
python examples/backtest_stock.py \
  --data ./data/your_stock_daily.csv \
  --lookback 400 \
  --mc_samples 8 \
  --up_thresh 0.002 \
  --conf_thresh 0.55 \
  --buy_fee 0.001 \
  --sell_fee 0.0015 \
  --device cuda:0
```
- 参数说明：  
  - `--data` 数据路径；`--lookback` 滑窗长度（≤ max_context）；  
  - `--mc_samples` MC 抽样次数；  
  - `--up_thresh` 触发做多的期望收益阈值；  
  - `--conf_thresh` 方向概率阈值；  
  - `--buy_fee/--sell_fee` 买/卖成本；  
  - `--device` 计算设备。  
  - 脚本内固定：`delay=1`、`allow_short=False`、`pred_len=1`。

### 30 分钟 ETF（T+0，多空）
```bash
python examples/backtest_etf.py \
  --data ./data/your_etf_30m.csv \
  --lookback 256 \
  --mc_samples 8 \
  --up_thresh 0.0008 \
  --down_thresh 0.0008 \
  --conf_thresh 0.55 \
  --buy_fee 0.001 \
  --sell_fee 0.0015 \
  --device cuda:0
```
- 参数说明：同上，增加  
  - `--down_thresh` 触发做空的期望收益阈值。  
  - 脚本固定：`delay=0`、`allow_short=True`、`bars_per_year=252*8`。

## 注意与扩展
- 现用“下一根 close 收益”驱动信号与执行，如需开盘成交、涨跌停过滤、滑点或固定手续费，可在 `backtest/engine.py` 调整。  
- MC 抽样为多次单样本预测，耗时随样本数线性增长，可调 `T/top_p/top_k` 控制随机性。  
- 可扩展多步预测或多资产批量：提升 `pred_len` 或使用 `KronosPredictor.predict_batch`。  
- 严格错误指标：`strict_wrong_up_loss/strict_wrong_down_loss` 为对应方向信号但亏损时的平均值（含成本）；`strict_wrong_*_count` 为次数。  
- 置信度分桶：调用 `run_backtest` 时传入 `prob_up/prob_down`（来自 `classify_mc_returns` 输出），可获得不同置信区间的平均收益。***
