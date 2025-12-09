import pandas as pd
import numpy as np
from typing import Tuple, List

from model import Kronos, KronosTokenizer, KronosPredictor


PRICE_COLS = ["open", "high", "low", "close", "volume", "amount"]


class PredictorRunner:
    """
    Thin wrapper to run KronosPredictor and generate Monte Carlo price samples.
    Uses repeated single-sample calls to obtain multiple stochastic trajectories.
    """

    def __init__(
        self,
        model_id: str = "NeoQuasar/Kronos-small",
        tokenizer_id: str = "NeoQuasar/Kronos-Tokenizer-base",
        device: str = "cpu",
        max_context: int = 512,
        T: float = 1.0,
        top_p: float = 0.9,
        top_k: int = 0,
    ) -> None:
        self.model_id = model_id
        self.tokenizer_id = tokenizer_id
        self.device = device
        self.max_context = max_context
        self.T = T
        self.top_p = top_p
        self.top_k = top_k

        self.tokenizer = KronosTokenizer.from_pretrained(self.tokenizer_id)
        self.model = Kronos.from_pretrained(self.model_id)
        self.tokenizer.eval()
        self.model.eval()
        self.predictor = KronosPredictor(
            self.model,
            self.tokenizer,
            device=self.device,
            max_context=self.max_context,
        )

    def predict_mc(
        self,
        df: pd.DataFrame,
        x_timestamp: pd.Series,
        y_timestamp: pd.Series,
        pred_len: int,
        mc_samples: int = 8,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate Monte Carlo samples of future prices.

        Returns:
            samples: shape (mc_samples, pred_len, len(PRICE_COLS))
            mean_path: shape (pred_len, len(PRICE_COLS))
        """
        samples: List[np.ndarray] = []
        for _ in range(mc_samples):
            pred_df = self.predictor.predict(
                df=df,
                x_timestamp=x_timestamp,
                y_timestamp=y_timestamp,
                pred_len=pred_len,
                T=self.T,
                top_k=self.top_k,
                top_p=self.top_p,
                sample_count=1,
                verbose=False,
            )
            samples.append(pred_df[PRICE_COLS].to_numpy(dtype=np.float32))

        samples_arr = np.stack(samples, axis=0)
        mean_path = samples_arr.mean(axis=0)
        return samples_arr, mean_path


def compute_step_return_samples(
    mc_samples: np.ndarray,
    last_close: float,
    step: int = 0,
) -> np.ndarray:
    """
    Compute return samples for a specific forecast step based on predicted close prices.

    Args:
        mc_samples: shape (mc_samples, pred_len, feat)
        last_close: last observed close price before forecast window
        step: which forecast step to use (default first step)

    Returns:
        ret_samples: shape (mc_samples,)
    """
    close_idx = PRICE_COLS.index("close")
    close_preds = mc_samples[:, step, close_idx]
    return close_preds / last_close - 1.0
