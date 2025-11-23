# src/tailrisk/backtest.py

from typing import Tuple
import numpy as np
import pandas as pd
from scipy.stats import chi2


def kupiec_test(
    returns: pd.Series,
    var_series: pd.Series,
    alpha: float = 0.99,
) -> Tuple[float, float, int, int]:
    """
    Kupiec (1995) Unconditional Coverage Test.

    H0: exceedance frequency = (1 - alpha)

    Parameters
    ----------
    returns : pd.Series
        Realized returns or P&L.
    var_series : pd.Series
        VaR time series (positive number).
    alpha : float
        Confidence level.

    Returns
    -------
    lr_uc : float
        Likelihood ratio statistic.
    p_value : float
        p-value for chi-square(1) distribution.
    n_exceed : int
        Number of exceedances.
    n_total : int
        Number of observations used.
    """
    aligned = pd.concat([returns, var_series], axis=1).dropna()
    r = aligned.iloc[:, 0]
    v = aligned.iloc[:, 1]

    # exceedance: realized loss > VaR
    # since returns are usually negative in tail, use -r > v
    exceed = (-r > v).astype(int)
    n_total = len(exceed)
    n_exceed = int(exceed.sum())

    if n_exceed == 0 or n_exceed == n_total:
        # edge cases; LR formula degenerates
        return np.nan, np.nan, n_exceed, n_total

    pi_hat = n_exceed / n_total
    pi0 = 1 - alpha

    log_l0 = (
        (n_exceed * np.log(pi0)) + ((n_total - n_exceed) * np.log(1 - pi0))
    )
    log_l1 = (
        (n_exceed * np.log(pi_hat)) + ((n_total - n_exceed) * np.log(1 - pi_hat))
    )

    lr_uc = -2 * (log_l0 - log_l1)
    p_value = 1 - chi2.cdf(lr_uc, df=1)
    return float(lr_uc), float(p_value), n_exceed, n_total


def christoffersen_test(
    returns: pd.Series,
    var_series: pd.Series,
    alpha: float = 0.99,
) -> Tuple[float, float]:
    """
    Christoffersen (1998) Conditional Coverage Test (independence part).

    Here we implement a simple 2x2 transition matrix test on exceedances.

    Returns
    -------
    lr_cc : float
        LR statistic (independence part only).
    p_value : float
        p-value with chi-square(1).
    """
    aligned = pd.concat([returns, var_series], axis=1).dropna()
    r = aligned.iloc[:, 0]
    v = aligned.iloc[:, 1]
    exceed = (-r > v).astype(int).values

    # Build transition counts N_ij
    n00 = n01 = n10 = n11 = 0
    for t in range(1, len(exceed)):
        i = exceed[t - 1]
        j = exceed[t]
        if i == 0 and j == 0:
            n00 += 1
        elif i == 0 and j == 1:
            n01 += 1
        elif i == 1 and j == 0:
            n10 += 1
        elif i == 1 and j == 1:
            n11 += 1

    n0 = n00 + n01
    n1 = n10 + n11

    if n0 == 0 or n1 == 0:
        return np.nan, np.nan

    pi01 = n01 / n0
    pi11 = n11 / n1
    pi  = (n01 + n11) / (n0 + n1)

    def safe_log(x):
        return np.log(x) if x > 0 else 0.0

    log_l0 = (
        n00 * safe_log(1 - pi) +
        n01 * safe_log(pi) +
        n10 * safe_log(1 - pi) +
        n11 * safe_log(pi)
    )

    log_l1 = (
        n00 * safe_log(1 - pi01) +
        n01 * safe_log(pi01) +
        n10 * safe_log(1 - pi11) +
        n11 * safe_log(pi11)
    )

    lr_cc = -2 * (log_l0 - log_l1)
    p_value = 1 - chi2.cdf(lr_cc, df=1)
    return float(lr_cc), float(p_value)
