
import pandas as pd
from .performance import annualized_vol, sharpe_ratio, sortino_ratio, drawdown_curve

def risk_summary(returns: pd.Series, rf=0.0, periods_per_year=252) -> pd.Series:
    nav = (1+returns).cumprod()
    dd = drawdown_curve(nav)
    out = {
        "Ann.Vol": annualized_vol(returns, periods_per_year),
        "Sharpe": sharpe_ratio(returns, rf, periods_per_year),
        "Sortino": sortino_ratio(returns, rf, periods_per_year),
        "Max Drawdown": dd.min()
    }
    return pd.Series(out)
