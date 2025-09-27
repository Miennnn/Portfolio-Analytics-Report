Portfolio Analytics Toolkit

A lightweight Python toolkit that ingests portfolio holdings/transactions/prices and produces performance, risk, attribution, a 60/40 vs Tilted backtest, and a Monte Carlo wealth projection—exported as a clean, portable HTML report with tables and charts.

✨ Features

Data ingest & validation (CSV): holdings, transactions, prices (with ffill/bfill & calendar alignment)

Performance: TWR/XIRR, rolling returns, CAGR

Risk: drawdown, annualized vol, Sharpe, Sortino

Attribution: Brinson-style allocation / selection / interaction (vs benchmark)

Backtest: compare 60/40 (SPY/AGG) vs your Tilted model (EU infra/financials + Japan + EM bonds), monthly rebalance with drift threshold

Monte Carlo: multi-path retirement/wealth projection with percentile bands

Reporting: single-file output/report.html + output/charts/*.png

Automation-ready: daily refresh script (macOS launchd, cron)

📁 Repo Layout
portfolio-analytics/
  data/
    holdings.csv
    transactions.csv
    prices.csv
  notebooks/
    analysis.ipynb
  src/
    __init__.py
    io.py
    performance.py
    risk.py
    attribution.py
    backtest.py
    monte_carlo.py
    report.py
  output/
    charts/
    report.html
  datacreating.py        # builds/updates prices.csv (demo data or yfinance)
  run_pipeline.py        # end-to-end: load → compute → plot → report
  requirements.txt
  README.md
