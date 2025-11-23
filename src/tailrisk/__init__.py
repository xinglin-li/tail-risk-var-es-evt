# src/tailrisk/__init__.py

from .returns import compute_log_returns
from .var_es_historical import historical_var, historical_es
from .var_es_parametric import (
    parametric_var_normal,
    parametric_es_normal,
    parametric_var_t,
    parametric_es_t,
)
from .evt_gpd import fit_gpd_mle, gpd_var_es
from .backtest import kupiec_test, christoffersen_test
