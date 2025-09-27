
import pandas as pd

REQUIRED_PRICE_COLS = {"date","ticker","close"}
REQUIRED_HOLD_COLS  = {"ticker","weight","asset_class","sector","region"}
REQUIRED_TX_COLS    = {"date","ticker","qty","price","fees"}

def _validate(df, required, name):
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{name} missing columns: {missing}")

def load_prices(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    _validate(df, REQUIRED_PRICE_COLS, "pricesUpdated.csv")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["ticker","date"])
    # fill missing closes by grp
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    df["close"] = (
    df.groupby("ticker", group_keys=False)["close"]
      .transform(lambda s: s.ffill().bfill())
)
    return df

def prices_wide(prices_long: pd.DataFrame) -> pd.DataFrame:
    wide = prices_long.pivot(index="date", columns="ticker", values="close").sort_index()
    # align calendars: forward fill then backfill edge
    wide = wide.ffill().bfill()
    return wide

def load_holdings(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    _validate(df, REQUIRED_HOLD_COLS, "holdings.csv")
    if not (abs(df["weight"].sum() - 1.0) < 1e-6):
        # normalize to 1
        df["weight"] = df["weight"] / df["weight"].sum()
    return df

def load_transactions(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    _validate(df, REQUIRED_TX_COLS, "transactions.csv")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date","ticker"])
    return df
