# src/tailrisk/evt_gpd.py

from dataclasses import dataclass
from typing import Tuple
import numpy as np
from scipy.stats import genpareto


@dataclass
class GPDParams:
    xi: float     # shape (tail index)
    beta: float   # scale
    u: float      # threshold
    p_exceed: float  # empirical exceedance probability


def fit_gpd_mle(returns: np.ndarray, u: float) -> GPDParams:
    """
    Fit GPD to negative tail: we assume losses L = -returns.

    We keep only exceedances over threshold u in the LOSS space:
        exceedances = L[L > u] - u

    Parameters
    ----------
    returns : np.ndarray
        Asset or portfolio returns.
    u : float
        Threshold in LOSS space (positive). For example, 95% or 97.5% empirical loss quantile.

    Returns
    -------
    GPDParams
    """
    losses = -returns  # convert to loss
    exceed = losses[losses > u] - u
    n = len(losses)
    n_exceed = len(exceed)
    if n_exceed == 0:
        raise ValueError("No exceedances above threshold u; choose a lower threshold.")

    # genpareto.fit returns (shape, loc, scale)
    # we fix loc = 0 by subtracting u above.
    xi, loc, beta = genpareto.fit(exceed, floc=0.0)

    p_exceed = n_exceed / n
    return GPDParams(xi=float(xi), beta=float(beta), u=float(u), p_exceed=float(p_exceed))


def gpd_var_es(
    gpd_params: GPDParams,
    alpha: float = 0.99,
) -> Tuple[float, float]:
    """
    Compute tail VaR and ES at level alpha using GPD (POT approach).

    We use the standard POT formula in LOSS space:

      VaR_alpha = u + (beta / xi) * [ ( ( (1 - alpha) / p_exceed )^{-xi} - 1 ) ],

    and ES:

      ES_alpha = VaR_alpha / (1 - xi) + (beta - xi * u) / (1 - xi),

    under xi < 1 for ES to be finite.

    Returns are converted back to "return space" as negative numbers:

      VaR_return  = - VaR_loss
      ES_return   = - ES_loss

    Finally, we report positive risk measures:
      VaR = - VaR_return > 0, ES = - ES_return > 0.

    So effectively VaR / ES here are positive loss numbers.
    """
    xi, beta, u, p_exceed = (
        gpd_params.xi,
        gpd_params.beta,
        gpd_params.u,
        gpd_params.p_exceed,
    )

    if p_exceed <= 0:
        raise ValueError("p_exceed must be positive.")

    # Loss-space VaR
    scale_factor = ((1 - alpha) / p_exceed) ** (-xi) - 1.0
    var_loss = u + (beta / xi) * scale_factor

    if xi >= 1:
        raise ValueError("xi >= 1 ⇒ ES is infinite under GPD. Model not appropriate.")

    es_loss = (var_loss + beta - xi * u) / (1 - xi)

    # Convert to positive risk numbers (VaR / ES)
    var = float(var_loss)
    es = float(es_loss)
    return var, es
