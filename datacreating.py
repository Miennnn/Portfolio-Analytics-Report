# datacreating.py — pull up to the latest completed US trading day (Python 3.12)
import os, shutil, time
import pandas as pd
import numpy as np
import yfinance as yf

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
OUT_PATH = os.path.join(DATA_DIR, "prices.csv")

# Map internal tickers -> public proxies (adjust as you like)
PROXY_MAP = {
    "SPY":     "SPY",   # S&P 500
    "AGG":     "AGG",   # US Aggregate Bond
    "EUINFRA": "IGF",   # Global Infrastructure (proxy for EU infra tilt)
    "EUFIN":   "EUFN",  # Europe Financials
    "JP":      "EWJ",   # Japan equities
    "EMB":     "EMB",   # EM USD Bond
}

START = pd.Timestamp("2022-01-01")

def latest_us_trading_date(now_et: pd.Timestamp | None = None) -> pd.Timestamp:
    """
    Return the most recent completed US trading day (US/Eastern midnight).
    - If weekday and after ~5:00pm ET, include today; else previous business day.
    - Weekends roll back to Friday.
    """
    if now_et is None:
        now_et = pd.Timestamp.now(tz="US/Eastern")

    if now_et.weekday() >= 5:  # Sat/Sun → previous business day
        end = (now_et - pd.tseries.offsets.BDay(1)).normalize()
    else:
        cutoff = now_et.normalize() + pd.Timedelta(hours=17)  # ~5pm ET buffer
        end = now_et.normalize() if now_et >= cutoff else (now_et - pd.tseries.offsets.BDay(1)).normalize()
    return end

def download_adj_close(symbols: list[str] | tuple[str, ...],
                       start_date: pd.Timestamp,
                       end_date_et: pd.Timestamp) -> pd.DataFrame:
    """
    Download Adjusted Close for symbols from start..end (inclusive) using yfinance.
    Pass date strings to avoid tz issues; yfinance end is exclusive so add +1 day.
    """
    s = start_date.strftime("%Y-%m-%d")
    e = (end_date_et + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    for attempt in range(3):
        try:
            df = yf.download(
                tickers=list(symbols), start=s, end=e,
                auto_adjust=False, progress=False, threads=True
            )
            if isinstance(df.columns, pd.MultiIndex):
                df = df["Adj Close"].copy()
            else:
                # single-symbol case uses regular columns
                sym = list(symbols)[0]
                if "Adj Close" in df.columns:
                    df = df.rename(columns={"Adj Close": sym})
            return df
        except Exception:
            if attempt == 2:
                raise
            time.sleep(0.8 * (attempt + 1))

def make_synth(n: int, mu: float = 0.08, sigma: float = 0.18, seed: int = 1234) -> np.ndarray:
    """Fallback: geometric-ish random walk starting at 100."""
    rng = np.random.default_rng(seed)
    dt = 1/252
    rets = rng.normal(mu*dt, sigma*np.sqrt(dt), size=n)
    return 100*np.cumprod(1+rets)

def main() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)

    end_et = latest_us_trading_date()
    print(f"[info] Using latest US trading day: {end_et.date()} (US/Eastern)")

    # 👇 drop timezone so both START and end are naive
    end_naive = end_et.tz_localize(None)

    # download (we pass strings inside download_adj_close, so tz doesn't matter)
    raw = download_adj_close(list(PROXY_MAP.values()), START, end_naive)

    # Align to business days using naive endpoints
    bidx = pd.bdate_range(START, end_naive, freq="C")
    raw = raw.reindex(bidx).ffill().bfill()

    # Build internal wide frame (our tickers)
    wide = pd.DataFrame(index=bidx)
    for i, (internal, proxy) in enumerate(PROXY_MAP.items()):
        if proxy in raw.columns and raw[proxy].notna().any():
            wide[internal] = raw[proxy]
        else:
            print(f"[warn] Missing {proxy}; generating synthetic series for {internal}.")
            wide[internal] = make_synth(len(bidx), seed=1000 + i)

    # Final clean & reshape to long (date,ticker,close)
    wide = wide.sort_index().ffill().bfill()
    prices_long = (
        wide.reset_index()
            .melt(id_vars="index", var_name="ticker", value_name="close")
            .rename(columns={"index": "date"})
    )
    prices_long["date"] = prices_long["date"].dt.strftime("%Y-%m-%d")

    # Backup then write
    if os.path.exists(OUT_PATH):
        bak = OUT_PATH.replace(".csv", ".bak")
        shutil.copy2(OUT_PATH, bak)
        print(f"[info] Backed up existing prices.csv -> {bak}")

    prices_long.to_csv(OUT_PATH, index=False)
    print(f"[ok] Wrote {OUT_PATH} ({len(bidx)} dates × {wide.shape[1]} tickers = {len(prices_long)} rows)")

if __name__ == "__main__":
    main()
