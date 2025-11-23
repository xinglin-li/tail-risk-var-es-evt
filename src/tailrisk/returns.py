# src/tailrisk/returns.py

from typing import Literal
import numpy as np
import pandas as pd


def compute_log_returns(
    prices: pd.Series,
    freq: Literal["daily", "intraday"] = "daily",
) -> pd.Series:
    """
    Compute log returns r_t = log(P_t / P_{t-1}).

    Parameters
    ----------
    prices : pd.Series
        Price time series (indexed by datetime).
    freq : {"daily", "intraday"}
        Only used for labeling / downstream logic if needed.

    Returns
    -------
    pd.Series
        Log returns, aligned with prices.index (first element is NaN).
    """
    prices = prices.sort_index()
    rets = np.log(prices / prices.shift(1))
    rets.name = "log_return"
    return rets.dropna()
