import numpy as np
import pandas as pd


def max_drawdown(cumulative_series: pd.Series) -> float:
    """
    Calcola il massimo drawdown.
    Ovvero, quanto avrei perso comprando nel momento peggiore e vendendo nel momento peggiore
    """

    running_max = cumulative_series.cummax() # picco massimo del portafoglio, fino a quel momento
    # sarebbe valore_attuale/picco_max = quanto mi è rimasto rispetto al momento migliore
    # -1 poiché vogliamo la perdita
    drawdown = cumulative_series / running_max - 1 # ogni giorno, 

    return float(drawdown.min()) # ritorno la caduta percentuale più profonda


def performance_metrics(
    returns_df: pd.DataFrame,
    risk_free_rate: float = 0.0, # tasso privo di rischio
    periods_per_year: int = 252 # numero di giorni di contrattazione standard in borsa
) -> pd.DataFrame:
    """
    Calcola metriche di performance annualizzate.
    """

    metrics = []

    for col in returns_df.columns: # per ogni strategia
        r = returns_df[col].dropna()

        if r.empty:
            continue
        
        # calcolo dell'equity line (interesse composto)
        # in parole povere, è il fattore moltiplicativo del capitale iniziale
        # giorno per giorno
        cumulative = (1 + r).cumprod()

        total_return = cumulative.iloc[-1] - 1 # rendimento netto
        n_days = len(r) # giorni effetttivi di borsa della simulazione
        n_years = n_days / periods_per_year # durata in anni di borsa

        # calcolo del Compound Annual Growth Rate
        '''
        Nota: non basta usare una media aritmetica.
        Infatti partendo, da 100 euro, se faccio +50% il primo anno
        e -50% il secondo, ho 75 euro, non 100.
        Il CAGR calcola quanto hai guadagnato/perso costantemente ogni anno.
        '''
        annual_return = (1 + total_return) ** (1 / n_years) - 1
        annual_volatility = r.std() * np.sqrt(periods_per_year)

        if annual_volatility != 0:
            # calcolo l'indice di Sharpe
            '''
            Il numeratore indica il rendimento (tolti ritorni sicuri, come titoli di stato)
            Il denominatore è la volatilità del portafoglio.
            Quindi: se guadagno tanto rischiando molto, avrò uno Sharpe basso.
            Al contrario, se guadagno molto rischiando poco, avrò uno sharpe alto.
            '''
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