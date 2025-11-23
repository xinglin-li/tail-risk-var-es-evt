# src/tailrisk/var_es_historical.py

import numpy as np
import pandas as pd


def historical_var(returns: pd.Series, alpha: float = 0.99) -> float:
    """
    Historical VaR at level alpha (e.g. 0.99).

    By convention, VaR is reported as a positive loss number:
        VaR_alpha = - quantile_{1 - alpha}(returns)

    Parameters
    ----------
    returns : pd.Series
        P&L or returns (can be single asset or portfolio).
    alpha : float
        Confidence level, e.g. 0.99.

    Returns
    -------
    float
        VaR at level alpha (positive number).
    """
    q = np.quantile(returns, 1 - alpha)
    return -float(q)


def historical_es(returns: pd.Series, alpha: float = 0.99) -> float:
    """
    Historical Expected Shortfall (ES) at level alpha.

    ES_alpha = - E[ r_t | r_t <= q_{1 - alpha} ].

    Parameters
    ----------
    returns : pd.Series
        P&L or returns.
    alpha : float
        Confidence level.

    Returns
    -------
    float
        ES at level alpha (positive number).
    """
    q = np.quantile(returns, 1 - alpha)
    tail = returns[returns <= q]
    return -float(tail.mean())
