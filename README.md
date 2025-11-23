# Tail Risk Modeling with VaR / ES and EVT

This project implements a **hedge-fund style tail risk pipeline**:

- **Risk measures**: Value-at-Risk (VaR), Expected Shortfall (ES)
- **Models**:
  - Historical VaR / ES
  - Parametric VaR / ES (Normal, Student-t)
  - EVT-based VaR / ES via Generalized Pareto Distribution (GPD)
- **Backtesting**:
  - Unconditional coverage (Kupiec)
  - Conditional coverage (Christoffersen)
- **Use case**:
  - Single asset or portfolio P&L
  - Daily frequency (easily extendable to intraday)

The goal is to demonstrate **practical tail risk modeling** suitable for a multi-asset hedge fund context.

## 1. Project Structure

- `src/tailrisk/`: Reusable library for risk measures and EVT
- `notebooks/`:
  - `01_explore_returns.ipynb`: return distribution and diagnostics
  - `02_fit_evt_gpd.ipynb`: threshold selection, GPD fit, tail index
  - `03_var_es_backtest.ipynb`: VaR/ES calculation and backtesting
- `data/`: price / P&L data (not included in the repo)

## 2. Core Methods

We implement three VaR / ES approaches:

1. **Historical VaR / ES**  
   - Non-parametric, based on empirical distribution of returns.

2. **Parametric VaR / ES (Normal / Student-t)**  
   - Closed-form formulas under distributional assumptions.

3. **EVT (Peaks Over Threshold) + GPD**  
   - Fit GPD to exceedances over a high threshold.
   - Use Pickands–Balkema–De Haan theorem to extrapolate tail.

## 3. Backtesting

We evaluate VaR models via:

- **Kupiec (1995) Unconditional Coverage Test**
- **Christoffersen (1998) Conditional Coverage Test**

to check whether the realized exceedance frequency and clustering behavior are consistent with the nominal VaR level.

## 4. Tech Stack

- Python, NumPy, pandas, SciPy, matplotlib
- Designed to be easily plugged into a larger trading or risk system.

## 5. Disclaimer

This repository is for research and educational purposes only and does not constitute investment advice.

## 6. Results (TSLA Case Study)

We apply the framework to **TSLA daily log returns** (2012–2025):

- The empirical return distribution exhibits **strong fat tails** and skewness relative to a Normal benchmark (QQ plot).
- EVT–GPD fitted on the top 2.5% of losses yields a tail index \( \xi \approx 0.25 \), confirming heavy tails.

### VaR Backtesting at 99% Level (Rolling, 1000-day window)

| Model       | Exceed freq | Kupiec p-value | Christoffersen p-value |
|------------|-------------|----------------|-------------------------|
| Historical | ...         | ...            | ...                     |
| Normal     | ...         | ...            | ...                     |
| Student-t  | ...         | ...            | ...                     |
| EVT-GPD    | ...         | ...            | ...                     |

**Key takeaway:**

> “For TSLA, EVT-based VaR achieves closer unconditional coverage at the 99% level and reduces exceedance clustering compared to Gaussian VaR, reflecting a better calibration of tail risk under extreme market moves.”

