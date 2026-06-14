import numpy as np
import pandas as pd
import networkx as nx


def correlation_to_distance(corr_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Trasforma la matrice di correlazione nella matrice delle distanze:

    d_ij = sqrt(2 * (1 - rho_ij))
    """

    arr = np.sqrt(2 * (1 - corr_matrix.to_numpy(dtype=float, copy=True)))
    np.fill_diagonal(arr, 0)  
    distance_matrix = pd.DataFrame(arr, index=corr_matrix.index, columns=corr_matrix.columns)

    return distance_matrix


def build_graph_from_distance(distance_matrix: pd.DataFrame) -> nx.Graph:
    """
    Costruisce un grafo completo pesato a partire dalla matrice delle distanze.
    Le distanze mancanti vengono ignorate.
    """

    graph = nx.Graph()
    symbols = distance_matrix.index.tolist()

    for symbol in symbols:
        graph.add_node(symbol)

    for i, symbol_i in enumerate(symbols):
        for j, symbol_j in enumerate(symbols):
            if j <= i:
                continue

            distance = distance_matrix.loc[symbol_i, symbol_j]

            if pd.notna(distance):
                graph.add_edge(
                    symbol_i,
                    symbol_j,
                    weight=float(distance)
                )

    return graph


def build_mst_from_matrix(
    data_matrix: pd.DataFrame,
    min_periods: int = 93
):
    """
    Costruisce:
    1. matrice di correlazione;
    2. matrice delle distanze;
    3. grafo completo;
    4. Minimum Spanning Tree.
    """

    data_matrix = data_matrix.dropna(axis=1, how="all")
    data_matrix = data_matrix.dropna(axis=0, how="all")

    corr_matrix = data_matrix.corr(
        method="pearson",
        min_periods=min_periods
    )

    distance_matrix = correlation_to_distance(corr_matrix)

    graph = build_graph_from_distance(distance_matrix)

    mst = nx.minimum_spanning_tree(
        graph,
        weight="weight",
        algorithm="kruskal"
    )

    if mst.number_of_edges() != mst.number_of_nodes() - 1:
        raise ValueError(
            "Il MST non è connesso. "
            "Prova a ridurre min_periods oppure controlla i missing data."
        )

    return mst, corr_matrix, distance_matrix


def symbolize_three_states(data_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Simbolizza ogni serie in 3 stati:

    1 = valore basso
    2 = valore centrale/stabile
    3 = valore alto

    Soglie:
    mean - std/2
    mean + std/2
    """

    symbols = pd.DataFrame(
        index=data_matrix.index,
        columns=data_matrix.columns,
        dtype=float
    )

    for col in data_matrix.columns:
        series = data_matrix[col]
        valid = series.dropna()

        if valid.empty:
            continue

        mean_i = valid.mean()
        std_i = valid.std()

        lower = mean_i - std_i / 2
        upper = mean_i + std_i / 2

        s = pd.Series(index=data_matrix.index, dtype=float)

        s.loc[series < lower] = 1
        s.loc[(series >= lower) & (series <= upper)] = 2
        s.loc[series > upper] = 3

        symbols[col] = s

    return symbols


def build_method1_mst(
    returns_window: pd.DataFrame,
    min_periods: int = 93
):
    """
    Metodo 1:
    MST costruito sui log-rendimenti.
    """

    return build_mst_from_matrix(
        returns_window,
        min_periods=min_periods
    )


def build_method2_mst(
    returns_window: pd.DataFrame,
    min_periods: int = 93
):
    """
    Metodo 2:
    MST costruito sui log-rendimenti simbolizzati.
    """

    symbols_returns = symbolize_three_states(returns_window)

    return build_mst_from_matrix(
        symbols_returns,
        min_periods=min_periods
    )


def build_method3_extended_symbols(
    symbols_returns: pd.DataFrame,
    symbols_money: pd.DataFrame
) -> pd.DataFrame:
    """
    Metodo 3:
    concatena simboli dei rendimenti e simboli del controvalore.

    Per dare peso doppio ai rendimenti, duplichiamo il blocco dei rendimenti:
    [rendimenti, rendimenti, controvalore]
    """

    returns_block_1 = symbols_returns.copy()
    returns_block_2 = symbols_returns.copy()
    money_block = symbols_money.copy()


    # mettiamo un indice alfanumerico invece delle date, altrimenti avrei conflitti di chiavi duplicate
    # perderemo le date, ma a noi interessa solo l'ordine
    returns_block_1.index = [
        f"r1_{i}" for i in range(len(returns_block_1))
    ]

    returns_block_2.index = [
        f"r2_{i}" for i in range(len(returns_block_2))
    ]

    money_block.index = [
        f"m_{i}" for i in range(len(money_block))
    ]

    extended = pd.concat(
        [returns_block_1, returns_block_2, money_block],
        axis=0
    )

    return extended


def build_method3_mst(
    returns_window: pd.DataFrame,
    money_window: pd.DataFrame,
    min_periods: int = 93
):
    """
    Metodo 3:
    MST costruito su rendimenti simbolizzati + controvalore simbolizzato.
    """

    # intersezione fra le colonne dei due dataframe e li riordino alfabeticamente
    common_symbols = sorted(
        set(returns_window.columns).intersection(money_window.columns)
    )

    # faccio l'intersezione anche per le date
    common_dates = returns_window.index.intersection(money_window.index)

    # sovrascrivo i dataframe originali con i sotto data-frame appena ottenuti
    returns_window = returns_window.loc[common_dates, common_symbols]
    money_window = money_window.loc[common_dates, common_symbols]

    symbols_returns = symbolize_three_states(returns_window)
    symbols_money = symbolize_three_states(money_window)

    # creo un nuovo dataframe "triplicato" con due volte i rendimenti e una volta il controvalore
    extended_symbols = build_method3_extended_symbols(
        symbols_returns=symbols_returns,
        symbols_money=symbols_money
    )

    # costruisco MST
    return build_mst_from_matrix(
        extended_symbols,
        min_periods=min_periods
    )


def symbolize_method4_returns_confirmed_by_money(
    returns_matrix: pd.DataFrame,
    money_matrix: pd.DataFrame
) -> pd.DataFrame:
    """
    Metodo 4:
    Il movimento del rendimento viene considerato significativo
    solo se confermato da controvalore superiore alla media.

    Simbolo 1:
        rendimento basso e controvalore >= media

    Simbolo 2:
        rendimento stabile oppure controvalore < media

    Simbolo 3:
        rendimento alto e controvalore >= media
    """

    common_symbols = sorted(
        set(returns_matrix.columns).intersection(money_matrix.columns)
    )

    common_dates = returns_matrix.index.intersection(money_matrix.index)

    returns_matrix = returns_matrix.loc[common_dates, common_symbols]
    money_matrix = money_matrix.loc[common_dates, common_symbols]

    symbols = pd.DataFrame(
        index=common_dates,
        columns=common_symbols,
        dtype=float
    )

    for col in common_symbols:
        r_i = returns_matrix[col]
        m_i = money_matrix[col]

        r_mean = r_i.mean(skipna=True)
        r_std = r_i.std(skipna=True)
        m_mean = m_i.mean(skipna=True)

        lower = r_mean - r_std / 2
        upper = r_mean + r_std / 2

        valid = r_i.notna() & m_i.notna()

        mask_down_confirmed = (
            valid &
            (m_i >= m_mean) &
            (r_i < lower)
        )

        mask_up_confirmed = (
            valid &
            (m_i >= m_mean) &
            (r_i > upper)
        )

        mask_stable = (
            valid &
            (
                (m_i < m_mean) |
                ((r_i >= lower) & (r_i <= upper))
            )
        )

        s = pd.Series(index=common_dates, dtype=float)

        s.loc[mask_down_confirmed] = 1
        s.loc[mask_stable] = 2
        s.loc[mask_up_confirmed] = 3

        symbols[col] = s

    return symbols


def build_method4_mst(
    returns_window: pd.DataFrame,
    money_window: pd.DataFrame,
    min_periods: int = 93
):
    """
    Metodo 4:
    MST costruito sui simboli ottenuti combinando rendimenti e controvalore.
    """

    symbols = symbolize_method4_returns_confirmed_by_money(
        returns_matrix=returns_window,
        money_matrix=money_window
    )

    return build_mst_from_matrix(
        symbols,
        min_periods=min_periods
    )


def compute_mst_centrality(mst: nx.Graph) -> pd.DataFrame:
    """
    Calcola misure di centralità del MST.
    """

    degree = dict(mst.degree())
    degree_centrality = nx.degree_centrality(mst)
    betweenness = nx.betweenness_centrality(mst, normalized=True)
    closeness = nx.closeness_centrality(mst, distance="weight")

    centrality = pd.DataFrame({
        "Symbol": list(mst.nodes()),
        "degree": [degree[node] for node in mst.nodes()],
        "degree_centrality": [
            degree_centrality[node] for node in mst.nodes()
        ],
        "betweenness": [
            betweenness[node] for node in mst.nodes()
        ],
        "closeness": [
            closeness[node] for node in mst.nodes()
        ],
    })

    centrality = centrality.sort_values(
        ["degree", "betweenness", "closeness"],
        ascending=False
    )

    return centrality