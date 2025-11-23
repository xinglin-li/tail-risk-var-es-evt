# src/tailrisk/var_es_parametric.py

from typing import Tuple
import numpy as np
from scipy.stats import norm, t


def fit_normal_params(returns: np.ndarray) -> Tuple[float, float]:
    mu = float(np.mean(returns))
    sigma = float(np.std(returns, ddof=1))
    return mu, sigma


def parametric_var_normal(mu: float, sigma: float, alpha: float = 0.99) -> float:
    """
    VaR under Normal assumption: r ~ N(mu, sigma^2).
    VaR_alpha = - (mu + sigma * z_{1-alpha})
    """
    z = norm.ppf(1 - alpha)
    return -(mu + sigma * z)


def parametric_es_normal(mu: float, sigma: float, alpha: float = 0.99) -> float:
    """
    ES under Normal: ES_alpha = - [ mu - sigma * φ(z) / (1 - alpha) ],
    where z = Φ^{-1}(1 - alpha).
    """
    z = norm.ppf(1 - alpha)
    phi = norm.pdf(z)
    es = mu - sigma * phi / (1 - alpha)
    return -es


def fit_t_params(returns: np.ndarray) -> Tuple[float, float, float]:
    """
    Fit Student-t via MLE using scipy.stats.t.fit.

    Returns
    -------
    df, loc, scale
    """
    df, loc, scale = t.fit(returns)
    return float(df), float(loc), float(scale)


def parametric_var_t(df: float, loc: float, scale: float, alpha: float = 0.99) -> float:
    """
    VaR under Student-t: r = loc + scale * T_df.
    """
    q = t.ppf(1 - alpha, df, loc=loc, scale=scale)
    return -float(q)


def parametric_es_t(df: float, loc: float, scale: float, alpha: float = 0.99) -> float:
    """
    ES for Student-t.

    ES_alpha = - E[r | r <= q_{1-alpha}].
    Closed form exists but is slightly more involved; here we implement
    the standard formula using the t-density.

    Reference:
      ES = - loc + scale * (f_t(q*) / ((1 - alpha) * (df - 1))) * (df + q*^2),
      where q* is the standard t-quantile (loc=0, scale=1).
    """
    q_star = t.ppf(1 - alpha, df)  # standard t
    density = t.pdf(q_star, df)
    es_standard = (density / ((1 - alpha) * (df - 1))) * (df + q_star**2)
    es = loc - scale * es_standard
    return -float(es)
