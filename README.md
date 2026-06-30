# Applicazione dei Minimum Spanning Tree all'analisi dei rendimenti azionari

Questo progetto applica quattro metodologie basate su Minimum Spanning Tree a un campione di titoli azionari italiani nel periodo 2015-2024. Gli MST vengono utilizzati per individuare i titoli più centrali nella rete di dipendenze tra rendimenti e per costruire strategie di portafoglio confrontate con benchmark pesati per capitalizzazione.

## Struttura del progetto

- `notebooks/`: contiene i notebook principali dell'analisi.
- `src/portfolio_backtester/`: contiene il pacchetto Python sviluppato per costruire MST, strategie di portafoglio, backtest e metriche.
- `data/processed/`: contiene i dati già puliti necessari per replicare l'analisi.
- `outputs/`: contiene grafici e tabelle finali.

## Metodi MST implementati

1. Metodo 1: MST sui log-rendimenti.
2. Metodo 2: MST sui log-rendimenti simbolizzati.
3. Metodo 3: MST su rendimenti simbolizzati e controvalore scambiato simbolizzato.
4. Metodo 4: MST su rendimenti simbolizzati validati dal controvalore scambiato.

## Strategie di portafoglio

Per ogni anno di investimento, gli MST vengono costruiti sui tre anni precedenti di dati. I titoli vengono selezionati in base alla centralità nel MST e confrontati con strategie pesate per capitalizzazione.

## Installazione

Da terminale, nella cartella principale del progetto:

```bash
pip install -e .
