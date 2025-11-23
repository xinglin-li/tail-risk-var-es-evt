# tail-risk-var-es-evt
tail-risk-var-es-evt/
├─ README.md
├─ pyproject.toml        # 或 setup.cfg，取决于你要不要打包
├─ requirements.txt
├─ data/
│   ├─ raw/              # 原始价格数据（csv/parquet）
│   └─ processed/        # 处理后的日度/分钟收益
├─ notebooks/
│   ├─ 01_explore_returns.ipynb
│   ├─ 02_fit_evt_gpd.ipynb
│   └─ 03_var_es_backtest.ipynb
├─ src/
│   └─ tailrisk/
│       ├─ __init__.py
│       ├─ config.py
│       ├─ data_loader.py
│       ├─ returns.py
│       ├─ var_es_historical.py
│       ├─ var_es_parametric.py
│       ├─ evt_gpd.py
│       ├─ var_es_evt.py
│       ├─ backtest.py
│       └─ plotting.py
└─ tests/
    ├─ test_returns.py
    ├─ test_var_es.py
    └─ test_evt_gpd.py
