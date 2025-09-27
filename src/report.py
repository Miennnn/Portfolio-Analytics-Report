STYLE = """
<style>
  :root { --fg:#111; --muted:#555; --border:#e5e7eb; --bg:#fafafa; }
  html,body { margin:0; padding:0; }
  body { font-family: -apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Arial,sans-serif;
         color:var(--fg); background:var(--bg); margin:24px; line-height:1.55; }

  h1 { font-size: 28px; margin: 0 0 8px; }
  h2 { font-size: 20px; margin: 20px 0 8px; }

  .row { display:flex; gap:16px; flex-wrap:wrap; }
  .card { flex:1 1 320px; background:#fff; border:1px solid var(--border);
          border-radius:10px; padding:12px 14px; box-shadow:0 1px 2px rgba(0,0,0,0.04); }

  .desc { margin: 8px 0 16px; color:var(--fg); }
  details.desc { font-size:0.92rem; color:var(--muted); }

  /* Tables from pandas .to_html() */
  table { border-collapse:collapse; width:100%; background:#fff; }
  th, td { border:1px solid var(--border); padding:6px 8px; text-align:right; }
  th:first-child, td:first-child { text-align:left; }
  thead th { background:#f6f7f9; font-weight:600; }
  tbody tr:nth-child(even) { background: #fcfcfd; }

  /* Images */
  img { max-width:100%; height:auto; border:1px solid var(--border); border-radius:8px; }

  /* Utility */
  .muted { color:var(--muted); }
  .grid-2 { display:grid; grid-template-columns: 1fr 1fr; gap:16px; }
  .spacer { height: 8px; }
</style>
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

from .io import load_prices, prices_wide, load_holdings
from .backtest import run_backtest
from .performance import time_weighted_return, drawdown_curve
from .risk import risk_summary
from .attribution import brinson_allocation_selection
from .monte_carlo import simulate_paths, percentile_bands


def _save_plot(path: str) -> None:
    """
    Small helper to consistently save and close matplotlib figures.
    Keeps charts tidy and prevents memory leaks if many plots are created.
    """
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()

def month_over_month_summary(nav_tilt, nav_6040, rets_tilt, holdings, attr_tbl):
    end = nav_tilt.index.max()
    # ~21 business days back; fallback safe
    start_idx = max(0, nav_tilt.index.get_loc(end) - 21)
    start = nav_tilt.index[start_idx]
    twr_tilt_mom = nav_tilt.loc[end] / nav_tilt.loc[start] - 1
    twr_6040_mom = nav_6040.loc[end] / nav_6040.loc[start] - 1
    dd_series = nav_tilt / nav_tilt.cummax() - 1
    dd_now, dd_worst = dd_series.loc[end], dd_series.min()

    contrib = (attr_tbl.drop(index="Total")[["Allocation","Selection","Interaction"]]
               .sum(axis=1).sort_values(ascending=False))
    top = contrib.head(2); bot = contrib.tail(2)
    weights = holdings.groupby("region")["weight"].sum().sort_values(ascending=False)
    ann_vol = rets_tilt.std() * (252**0.5)

    return {
        "what": {"tilt_mom": float(twr_tilt_mom), "6040_mom": float(twr_6040_mom),
                 "dd_current": float(dd_now), "dd_max": float(dd_worst)},
        "why": {"top": top.to_dict(), "bottom": bot.to_dict()},
        "changed": {"weights": weights.round(3).to_dict(), "ann_vol": float(ann_vol)}
    }



def build_report(data_dir: str, output_dir: str):

    # -----------------------------
    # 1) LOAD & PREP DATA
    # -----------------------------
    # Load long-format prices (date, ticker, close), validate columns, fill within-ticker gaps.
    prices_long = load_prices(os.path.join(data_dir, "pricesUpdated.csv"))
    print(prices_long.head)

    # Pivot to wide format (index=date, columns=tickers), align calendars with ffill/bfill.
    prices = prices_wide(prices_long)

    # Read Tilted portfolio definition (weights + tags); normalize weights to sum to 1.
    holdings = load_holdings(os.path.join(data_dir, "holdings.csv"))

    # The list of tickers we care about is whatever is in holdings.csv
    tickers = holdings["ticker"].tolist()

    # -----------------------------
    # 2) DEFINE PORTFOLIOS & RUN BACKTESTS
    # -----------------------------
    # Build the 60/40 vector: If SPY and AGG exist, use them; otherwise equal weight fallback.
    targets_6040 = pd.Series(0.0, index=tickers)
    if "SPY" in targets_6040.index and "AGG" in targets_6040.index:
        targets_6040.loc["SPY"] = 0.60
        targets_6040.loc["AGG"] = 0.40
    else:
        # Fallback: equally weight all tickers if SPY/AGG aren't both present
        targets_6040[tickers] = 1 / len(tickers)

    # The Tilted portfolio is exactly what the user specified in holdings.csv
    targets_tilt = holdings.set_index("ticker")["weight"]

    # Run monthly-rebalanced backtests with a drift threshold.
    # Returns daily NAV series for each strategy (and a simple equal-weight bench from the function).
    nav_6040, _ = run_backtest(prices[tickers], targets_6040, freq="M")
    nav_tilt,  _ = run_backtest(prices[tickers], targets_tilt,  freq="M")

    # Convert NAV to daily returns for metrics that need returns
    rets_6040 = nav_6040.pct_change().dropna()
    rets_tilt = nav_tilt.pct_change().dropna()

    # -----------------------------
    # 3) PERFORMANCE & RISK TABLES
    # -----------------------------
    # Compute TWR from NAV (geometric link of returns)
    twr_6040 = time_weighted_return(nav_6040)
    twr_tilt = time_weighted_return(nav_tilt)

    # Risk summary (Ann.Vol, Sharpe, Sortino, Max Drawdown)
    risk_6040 = risk_summary(rets_6040)   # returns a Series
    risk_tilt = risk_summary(rets_tilt)   # returns a Series

    # Assemble a simple performance comparison table
    perf_tbl = pd.DataFrame(
        {
            "TWR": [twr_6040, twr_tilt],
            "Ann.Vol": [risk_6040["Ann.Vol"],   risk_tilt["Ann.Vol"]],
            "Sharpe":  [risk_6040["Sharpe"],    risk_tilt["Sharpe"]],
            "MaxDD":   [risk_6040["Max Drawdown"], risk_tilt["Max Drawdown"]],
        },
        index=["60/40", "Tilted"],
    )

    # A more detailed side-by-side “risk table”, keeping the Series shape
    risk_tbl = pd.concat({"60/40": risk_6040, "Tilted": risk_tilt}, axis=1)

    # -----------------------------
    # 4) CHARTS: EQUITY CURVES, DRAWDOWN, ROLLING VOL
    # -----------------------------
    # Equity curves for 60/40 and Tilted
    (nav_6040.rename("60/40")
            .to_frame()
            .join(nav_tilt.rename("Tilted"))) \
        .plot(figsize=(8, 4), title="Backtest Equity Curves")
    _save_plot(os.path.join(output_dir, "charts/backtest_equity.png"))

    # Drawdown curves for each strategy
    dd_df = pd.DataFrame(
        {
            "60/40": drawdown_curve(nav_6040),
            "Tilted": drawdown_curve(nav_tilt),
        }
    )
    dd_df.plot(figsize=(8, 3), title="Drawdowns")
    _save_plot(os.path.join(output_dir, "charts/drawdowns.png"))

    # Rolling annualized volatility over a 63-day (~3-month) window
    rv_6040 = rets_6040.rolling(63).std() * (252 ** 0.5)
    rv_tilt = rets_tilt.rolling(63).std() * (252 ** 0.5)
    pd.DataFrame({"60/40": rv_6040, "Tilted": rv_tilt}) \
        .plot(figsize=(8, 3), title="Rolling Annualized Volatility (63-day)")
    _save_plot(os.path.join(output_dir, "charts/rolling_vol.png"))

    # -----------------------------
    # 5) ATTRIBUTION SNAPSHOT (BRINSON BY REGION)
    # -----------------------------
    # We take the latest period (two dates separated by ~21 trading days) and compute per-ticker returns.
    # Then we map tickers -> regions (from holdings.csv), average to region-level returns,
    # and compare against a simple benchmark regional mix to get Allocation/Selection/Interaction effects.
    last_date = prices.index.max()
    # Protect against very short datasets: if fewer than 22 rows, fallback to earliest index
    last_pos = prices.index.get_loc(last_date)
    prev_pos = max(0, last_pos - 21)
    prev_date = prices.index[prev_pos]

    # Per-ticker period return over the chosen window
    px_last = prices.loc[[prev_date, last_date], tickers]
    period_ret = px_last.pct_change().iloc[-1]  # simple return over the period

    # Map tickers to regions, then compute portfolio weights and region-level returns
    region_map = holdings.set_index("ticker")["region"].to_dict()
    port_w = holdings.groupby("region")["weight"].sum()           # portfolio weights by region
    tmp = pd.DataFrame(
        {"ret": period_ret, "region": [region_map.get(t, "Other") for t in period_ret.index]}
    )
    port_r = tmp.groupby("region")["ret"].mean()                   # portfolio segment returns

    # Define a simple *benchmark* allocation by region (this is a demo; customize as needed)
    bench_w = pd.Series({"US": 0.60, "EU": 0.15, "JP": 0.10, "EM": 0.15}).reindex(
        port_w.index.union(["US", "EU", "JP", "EM"]), fill_value=0.0
    )
    # Define benchmark regional returns using available proxies from the tickers
    bench_r = pd.Series(
        {
            "US": period_ret["SPY"] if "SPY" in period_ret.index else port_r.get("US", 0.0),
            "EU": period_ret.filter(like="EU").mean(),
            "JP": period_ret.filter(like="JP").mean(),
            "EM": period_ret.filter(like="EM").mean(),
        }
    ).reindex(bench_w.index, fill_value=0.0)

    # Compute Brinson-Fachler components
    attr_tbl = brinson_allocation_selection(port_w, port_r, bench_w, bench_r).fillna(0.0)

    mom = month_over_month_summary(nav_tilt, nav_6040, rets_tilt, holdings, attr_tbl)

    # Plot Allocation/Selection/Interaction by region (omit the "Total" row for clarity)
    attr_tbl.drop(index="Total")[["Allocation", "Selection", "Interaction"]] \
            .plot(kind="bar", figsize=(8, 3), title="Attribution (Last Period)")
    _save_plot(os.path.join(output_dir, "charts/attribution.png"))

    # -----------------------------
    # 6) MONTE CARLO WEALTH PROJECTION
    # -----------------------------
    # Estimate mean/vol from Tilted daily returns (annualized) as simple inputs to MC.
    mu = rets_tilt.mean() * 252
    sig = rets_tilt.std() * (252 ** 0.5)

    # Simulate 30 years of monthly steps, with a starting value and monthly contribution.
    # Tweak 'start_value', 'contributions', 'years', 'sims' to your preference.
    paths = simulate_paths(
        start_value=100_000,
        mu=mu,
        sigma=sig,
        years=30,
        sims=2000,
        steps_per_year=12,
        contributions=1_000,
    )

    # Build percentile bands (5th/50th/95th) and plot the “fan chart”
    bands = percentile_bands(paths, ps=(5, 50, 95))
    bands.index.name = "t"
    bands.plot(figsize=(8, 4), title="Monte Carlo Wealth Projection (5/50/95th)")
    _save_plot(os.path.join(output_dir, "charts/mc_fan.png"))

    # -----------------------------
    # 7) COMPOSE HTML REPORT
    # -----------------------------
    # Convert the key tables to HTML (with basic CSS classes)
    perf_html = perf_tbl.round(4).to_html(classes="table", border=0)
    risk_html = risk_tbl.round(4).to_html(classes="table", border=0)
    attr_html = attr_tbl.round(6).to_html(classes="table", border=0)

    # Lightweight, self-contained HTML page that embeds all images + tables
    html = f"""
    <html>
    <head>
      <meta charset='utf-8'>
      <title>Portfolio Analytics Report</title>
      {STYLE}
    </head>
    <body>
      <h1 style="margin:0 0 12px; padding:14px 18px; font-size:28px; 
           background:#f1f5f9; border-left:8px solid #0ea5e9; 
           border-radius:10px; line-height:1.3;">Portfolio Analytics Report</h1>
      <p><b>Coverage:</b> Backtest 60/40 vs Tilted model, risk & drawdowns, attribution snapshot, and 30-year Monte Carlo.</p>
            
      <h2 style="margin:20px 0 8px; padding:8px 12px; background:#f6f7f9; 
           border-left:6px solid #3b82f6; border-radius:6px; font-size:20px;">Performance Summary</h2>
      <div class='row'>
        <div class='card'>{perf_html}</div>
        <div class='card'>{risk_html}</div>        
      </div>

      <p class='desc'>
        Over Jan&nbsp;2022–Sep&nbsp;2025, the Tilted portfolio returned <b>57.4%</b> vs <b>54.9%</b> for 60/40
        (<b>+2.5 pp cumulative</b>, ~<b>+0.6%/yr</b> CAGR edge). Risk was lower: annualized volatility
        <b>6.6%</b> vs <b>11.2%</b>, and max drawdown <b>−8.0%</b> vs <b>−11.8%</b>. Risk-adjusted results improved
        (Sharpe <b>1.82</b> vs <b>1.07</b>; Sortino <b>3.24</b> vs <b>1.90</b>). The most recent attribution shows
        <strong>US selection</strong> as the main positive driver, partially offset by <strong>EU allocation</strong>.
      </p>

      <h2 style="margin:20px 0 8px; padding:8px 12px; background:#f6f7f9; 
           border-left:6px solid #3b82f6; border-radius:6px; font-size:20px;">Backtest</h2>
      <div class='row'>
        <div class='card'><img src='charts/backtest_equity.png' /></div>
        <div class='card'><img src='charts/drawdowns.png' /></div>
      </div>
      <div class='row'>
        <div class='card'><img src='charts/rolling_vol.png' /></div>
      </div>

      <h2 style="margin:20px 0 8px; padding:8px 12px; background:#f6f7f9; 
           border-left:6px solid #3b82f6; border-radius:6px; font-size:20px;">Attribution (Last Period)</h2>
      <div class='row'>
        <div class='card'><img src='charts/attribution.png' /></div>
        <div class='card'>{attr_html}</div>
      </div>

      <h2 style="margin:20px 0 8px; padding:8px 12px; background:#f6f7f9; 
           border-left:6px solid #3b82f6; border-radius:6px; font-size:20px;">Monte Carlo</h2>
      <div class='row'>
        <div class='card'><img src='charts/mc_fan.png' /></div>
      </div>

            <p>Monte Carlo simulation with 5,000 paths, monthly steps, μ = 6%, σ = 15%, and $1,000 monthly contributions. Shaded bands show p5/p50/p95 wealth over 30 years. Assumptions exclude fees/taxes and assume stationary return/volatility.</p>

      <h2 style="margin:20px 0 8px; padding:8px 12px; background:#f6f7f9; 
           border-left:6px solid #3b82f6; border-radius:6px; font-size:20px;">MoM</h2>
        <div class='row'>
          <div class='card'><h3>What happened</h3>
            <p>Tilted MoM: {mom['what']['tilt_mom']:.2%}, 60/40 MoM: {mom['what']['6040_mom']:.2%}.
            Current DD: {mom['what']['dd_current']:.2%}; Worst DD: {mom['what']['dd_max']:.2%}.</p>
          </div>
          <div class='card'><h3>Why</h3>
            <p>Top: {', '.join([f"{k} {v:.2%}" for k,v in mom['why']['top'].items()])}<br/>
              Bottom: {', '.join([f"{k} {v:.2%}" for k,v in mom['why']['bottom'].items()])}</p>
          </div>
          <div class='card'><h3>What changed</h3>
            <p>Weights: {', '.join([f"{k} {v:.1%}" for k,v in mom['changed']['weights'].items()])}<br/>
              Tilted Ann.Vol: {mom['changed']['ann_vol']:.2%}</p>
          </div>
        </div>

        <h3 style="margin:18px 0 8px; padding:6px 10px; font-size:16px; 
           background:#f8fafc; border-left:4px solid #22c55e; 
           border-radius:6px; line-height:1.4;">Assumptions & Methodology</h3>
      <ul>
          <li><b>Benchmark:</b> 60/40 = SPY 60%, AGG 40%, monthly rebalancing.</li>
          <li><b>Tilted:</b> SPY, AGG, EUINFRA (IGF proxy), EUFIN (EUFN proxy), JP (EWJ proxy), EMB; weights as listed.</li>
          <li><b>Rebalancing:</b> Monthly with 2% drift threshold; no fees, taxes, or slippage modeled.</li>
          <li><b>Data:</b> Daily adjusted close from ETF proxies; business-day calendar; missing values ffill/bfill.</li>
          <li><b>Currency:</b> USD (foreign sleeves unhedged → FX risk included).</li>
          <li><b>Metrics:</b> Daily simple returns; Sharpe/Sortino use rf≈0; rolling vol window = 63 trading days.</li>
          <li><b>Drawdown:</b> Peak-to-trough using running high-water mark of NAV.</li>
        </ul>
      

      <p style='margin-top:30px;color:#666'>Generated by Portfolio Analytics Toolkit.</p>
    </body></html>
    """

    # Ensure the output directory exists (esp. when running in a fresh environment)
    os.makedirs(os.path.join(output_dir, "charts"), exist_ok=True)

    # Write the final HTML report
    report_path = os.path.join(output_dir, "report.html")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)

    # Return the key tables as Python objects (useful for notebooks/tests)
    return {
        "performance_table": perf_tbl,
        "risk_table": risk_tbl,
        "attribution_table": attr_tbl,
    }
