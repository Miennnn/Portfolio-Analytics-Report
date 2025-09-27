
import pandas as pd
import numpy as np

def time_weighted_return(nav_series: pd.Series) -> float:
    rets = nav_series.pct_change().dropna()
    return (1 + rets).prod() - 1

def money_weighted_return(cashflows: pd.Series) -> float:
    """
    XIRR on irregular cashflows. cashflows: pd.Series indexed by datetime with amounts (positive = inflow)
    Uses Newton's method.
    """
    dates = (cashflows.index - cashflows.index.min()).days / 365.25
    amounts = cashflows.values

    def npv(rate):
        return np.sum(amounts / (1+rate)**dates)

    r = 0.1
    for _ in range(100):
        # derivative
        denom = (1+r)**(dates+1)
        dnpv = -np.sum(dates * amounts / denom)
        val = npv(r)
        if abs(val) < 1e-10:
            break
        r_new = r - val / dnpv if dnpv != 0 else r
        if abs(r_new - r) < 1e-9:
            r = r_new
            break
        r = r_new
    return r

def rolling_return(price: pd.Series, window=63):
    return price.pct_change(periods=window)

def drawdown_curve(nav: pd.Series) -> pd.Series:
    roll_max = nav.cummax()
    dd = nav / roll_max - 1.0
    return dd

def annualized_vol(returns: pd.Series, periods_per_year=252) -> float:
    return returns.std() * np.sqrt(periods_per_year)

def sharpe_ratio(returns: pd.Series, rf=0.0, periods_per_year=252) -> float:
    excess = returns - rf/periods_per_year
    vol = annualized_vol(excess, periods_per_year)
    return (excess.mean()*periods_per_year) / vol if vol != 0 else np.nan

def sortino_ratio(returns: pd.Series, rf=0.0, periods_per_year=252) -> float:
    excess = returns - rf/periods_per_year
    downside = excess[excess < 0]
    dd_vol = downside.std() * np.sqrt(periods_per_year)
    return (excess.mean()*periods_per_year) / dd_vol if dd_vol != 0 else np.nan
