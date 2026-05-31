import numpy as np
import pandas as pd


def max_drawdown(cumulative_series: pd.Series) -> float:
    """
    Calcola il massimo drawdown.
    """

    running_max = cumulative_series.cummax()
    drawdown = cumulative_series / running_max - 1

    return float(drawdown.min())


def performance_metrics(
    returns_df: pd.DataFrame,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> pd.DataFrame:
    """
    Calcola metriche di performance annualizzate.
    """

    metrics = []

    for col in returns_df.columns:
        r = returns_df[col].dropna()

        if r.empty:
            continue

        cumulative = (1 + r).cumprod()

        total_return = cumulative.iloc[-1] - 1
        n_days = len(r)
        n_years = n_days / periods_per_year

        annual_return = (1 + total_return) ** (1 / n_years) - 1
        annual_volatility = r.std() * np.sqrt(periods_per_year)

        if annual_volatility != 0:
            sharpe = (
                annual_return - risk_free_rate
            ) / annual_volatility
        else:
            sharpe = np.nan

        metrics.append({
            "strategy": col,
            "total_return": total_return,
            "annual_return": annual_return,
            "annual_volatility": annual_volatility,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_drawdown(cumulative),
            "n_days": n_days
        })

    metrics_df = pd.DataFrame(metrics)

    return metrics_df