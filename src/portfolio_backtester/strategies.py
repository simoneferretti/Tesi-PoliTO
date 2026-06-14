import numpy as np
import pandas as pd


def normalize_weights(weights: pd.Series) -> pd.Series:
    """
    Normalizza una serie di pesi, rimuovendo valori mancanti,
    infiniti o non positivi.
    Lo useremo, ad esempio per normalizzare valori come il grado di un nodo,
    usato nella costruzione di un portafoglio.
    """

    weights = weights.astype(float)
    weights = weights.replace([np.inf, -np.inf], np.nan)
    weights = weights.dropna()
    weights = weights[weights > 0]

    total = weights.sum()

    if total <= 0:
        return pd.Series(dtype=float)

    return weights / total


def get_latest_market_cap(
    market_cap_matrix: pd.DataFrame,
    date
) -> pd.Series:
    """
    Prende l'ultima capitalizzazione disponibile prima o alla data indicata.
    Utilizzata prima di investire.
    """

    date = pd.Timestamp(date) # prendiamo la data

    available_dates = market_cap_matrix.index[
        market_cap_matrix.index <= date
    ] # selezioniamo le date non successive

    if len(available_dates) == 0:
        return pd.Series(dtype=float)

    last_date = available_dates.max() # prendo la data più recente

    caps = market_cap_matrix.loc[last_date].copy() #estraiamo la tupla relativa alla data selezionata
    caps = caps.replace([np.inf, -np.inf], np.nan)
    caps = caps.dropna()
    caps = caps[caps > 0]

    return caps


def weights_by_degree(
    centrality: pd.DataFrame,
    x: int
) -> pd.Series:
    """
    Seleziona i primi x titoli per degree e pesa proporzionalmente al degree.
    """

    selected = centrality.head(x).copy() #seleziono i primi x titoli per centralità

    weights = selected.set_index("Symbol")["degree"].astype(float) #creo una tabella azienda-degree

    return normalize_weights(weights)


def weights_by_market_cap_for_selected_symbols(
    selected_symbols: list[str],
    caps: pd.Series
) -> pd.Series:
    """
    Pesa per capitalizzazione solo i titoli selezionati.
    """

    # controlliamo che i titoli scelti in base al MST
    # siano nel df delle capitalizzazioni 
    available_symbols = [
        symbol for symbol in selected_symbols
        if symbol in caps.index
    ]

    if len(available_symbols) == 0:
        return pd.Series(dtype=float)

    weights = caps.loc[available_symbols]

    return normalize_weights(weights)


def weights_top_market_cap(
    caps: pd.Series,
    x: int
) -> pd.Series:
    """
    Seleziona i primi x titoli per capitalizzazione
    e pesa proporzionalmente alla capitalizzazione.
    """

    top_caps = caps.sort_values(ascending=False).head(x)

    return normalize_weights(top_caps)