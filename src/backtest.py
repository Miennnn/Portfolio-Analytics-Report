
import pandas as pd
import numpy as np

def rebalance(weights: pd.Series, targets: pd.Series, threshold=0.02):
    drift = (weights - targets).abs()
    if (drift > threshold).any():
        return targets.copy()
    return weights

def _returns_from_prices(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change().fillna(0.0)

def run_backtest(price_df: pd.DataFrame, target_weights: pd.Series, freq="M", threshold=0.02):
    """
    price_df: wide prices
    target_weights: pd.Series summing to 1 (index tickers subset of price_df.columns)
    Monthly rebal; hold cash 0.
    """
    prices = price_df[target_weights.index].copy()
    rets = _returns_from_prices(prices)

    # Rebalance dates
    rebal_dates = prices.resample(freq).last().index
    w = target_weights.copy()
    port_nav = []
    curr_nav = 1.0

    for t in range(len(prices)):
        dt = prices.index[t]
        if dt in rebal_dates:
            w = rebalance(w, target_weights, threshold)
        r = (w * rets.iloc[t]).sum()
        curr_nav *= (1 + r)
        port_nav.append(curr_nav)

    nav_series = pd.Series(port_nav, index=prices.index, name="Portfolio")
    bench = (1 + rets.mean(axis=1)).cumprod()  # naive equal-weight benchmark for demo
    return nav_series, bench.rename("Benchmark")
