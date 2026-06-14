from dataclasses import dataclass
from typing import Dict, Mapping

import numpy as np
import pandas as pd


@dataclass
class BacktestConfig:
    """
    Parametri configurabili del backtest.
    """

    initial_capital: float = 100_000.0
    transaction_fixed_cost: float = 5.0
    transaction_rate: float = 0.0
    capital_gain_tax_rate: float = 0.26
    min_trade_value: float = 1.0
    use_tax_loss_carryforward: bool = True
    fill_missing_returns_with_zero: bool = True


class PortfolioBacktester:
    """
    Backtester giornaliero per portafogli azionari a lungo termine.

    Il modello considera:
    - ribilanciamenti in date prefissate;
    - costo fisso per ordine;
    - costo percentuale opzionale;
    - tassazione sulle plusvalenze realizzate;
    - compensazione delle minusvalenze realizzate.

    Ogni posizione mantiene:
    - valore corrente di mercato;
    - costo fiscale residuo.
    """

    def __init__(
        self,
        returns_simple: pd.DataFrame,
        config: BacktestConfig | None = None
    ):
        self.returns = returns_simple.sort_index().copy()
        self.config = config or BacktestConfig()
        self.reset()

    def reset(self):
        """
        Resetta lo stato del backtest.
        """

        self.cash = float(self.config.initial_capital)
        self.positions: Dict[str, Dict[str, float]] = {}
        self.loss_carryforward = 0.0

        self.trade_log = []
        self.tax_log = []
        self.daily_log = []

    def portfolio_value(self) -> float:
        """
        Valore totale del portafoglio:
        liquidità + valore delle posizioni.
        """

        positions_value = sum(
            position["value"]
            for position in self.positions.values()
        )

        # liquidità + valore di mercato dei titoli
        return self.cash + positions_value

    def _transaction_cost(self, trade_value: float) -> float:
        """
        Calcola il costo di transazione di un ordine.
        """

        if trade_value < self.config.min_trade_value:
            return 0.0

        # costo fisso + costo percentuale 
        return (
            self.config.transaction_fixed_cost
            + self.config.transaction_rate * trade_value
        )

    def _apply_daily_returns(self, date: pd.Timestamp):
        """
        Applica i rendimenti giornalieri alle posizioni aperte.
        """

        if date not in self.returns.index:
            return

        daily_returns = self.returns.loc[date]
        
        # Aggiorno il valore di ogni posizione in base al rendimento giornaliero
        for symbol, position in list(self.positions.items()):
            # solo se il rendimento è disponibile
            r = daily_returns.get(symbol, np.nan)

            if pd.isna(r):
                if self.config.fill_missing_returns_with_zero:
                    r = 0.0
                else:
                    continue

            # aggiorno il valore di mercato della posizione
            position["value"] *= 1.0 + float(r) 

    def _process_tax_on_sale(
        self,
        date: pd.Timestamp,
        symbol: str,
        sale_value: float,
        basis_sold: float, # quanto ho pagato per il titolo all'acquisto
        transaction_cost: float
    ) -> tuple[float, float]:
        """
        Calcola plusvalenza/minusvalenza realizzata e tassa dovuta.

        La plusvalenza viene calcolata come:

        ricavo netto di vendita - costo fiscale venduto

        dove:

        ricavo netto = sale_value - transaction_cost

        Alla fine ritorna:
        realized_gain_after_costs : float
            Plusvalenza o minusvalenza realizzata.
        tax_paid : float
            Tassa pagata.
        """

        # calcolo il ricavo netto di vendita
        net_sale_proceeds = sale_value - transaction_cost

        # calcolo la plusvalenza/minusvalenza realizzata
        realized_gain = net_sale_proceeds - basis_sold

        tax_paid = 0.0

        # Se c'è una plusvalenza, calcolo la tassa dovuta considerando eventuali minusvalenze pregresse.
        if realized_gain > 0:
            if self.config.use_tax_loss_carryforward:
                # prendo il valore minimo fra la minusvalenza residua
                # e la plusvalenza realizzata
                offset = min(
                    self.loss_carryforward,
                    realized_gain
                )
                
                # tassabile
                taxable_gain = realized_gain - offset
                # aggiorno la minusvalenza residua
                self.loss_carryforward -= offset
            else:
                taxable_gain = realized_gain

            tax_paid = taxable_gain * self.config.capital_gain_tax_rate

        elif realized_gain < 0:
            if self.config.use_tax_loss_carryforward:
                # aggiorno la minusvalenza residua
                self.loss_carryforward += -realized_gain

        self.tax_log.append({
            "date": date,
            "symbol": symbol,
            "sale_value": sale_value,
            "transaction_cost": transaction_cost,
            "basis_sold": basis_sold,
            "realized_gain_after_costs": realized_gain,
            "tax_paid": tax_paid,
            "loss_carryforward_after": self.loss_carryforward
        })

        return realized_gain, tax_paid

    def _sell(
        self,
        date: pd.Timestamp,
        symbol: str,
        sale_value: float
    ):
        """
        Vende una quota della posizione.
        """

        if symbol not in self.positions:
            return

        position = self.positions[symbol]
        current_value = position["value"]

        if current_value <= 0:
            return

        # non posso vendere più del valore corrente della posizione (per gestire errori)
        sale_value = min(float(sale_value), current_value)

        # se il valore di vendita è troppo basso (<1€), ignoro l'operazione
        if sale_value < self.config.min_trade_value:
            return

        # commissione
        transaction_cost = self._transaction_cost(sale_value)

        # calcolo percentuale venduta rispetto al valore corrente
        basis_sold = position["cost_basis"] * (
            sale_value / current_value
        )

        realized_gain, tax_paid = self._process_tax_on_sale(
            date=date,
            symbol=symbol,
            sale_value=sale_value,
            basis_sold=basis_sold,
            transaction_cost=transaction_cost
        )

        # aggiorno la liquidità considerando ricavo di vendita, costi e tasse
        self.cash += sale_value - transaction_cost - tax_paid

        # scalo il valore del portafoglio
        position["value"] -= sale_value
        # aggiorno il costo d'acquisto residuo
        position["cost_basis"] -= basis_sold

        if position["value"] <= self.config.min_trade_value:
            del self.positions[symbol]

        # aggiungo il trade allo storico
        self.trade_log.append({
            "date": date,
            "symbol": symbol,
            "side": "SELL",
            "trade_value": sale_value,
            "transaction_cost": transaction_cost,
            "basis_sold": basis_sold,
            "realized_gain_after_costs": realized_gain,
            "tax_paid": tax_paid,
            "cash_after": self.cash,
            "portfolio_value_after": self.portfolio_value()
        })

    def _buy(
        self,
        date: pd.Timestamp,
        symbol: str,
        buy_value: float
    ):
        """
        Compra una posizione.
        """

        buy_value = float(buy_value)

        if buy_value < self.config.min_trade_value:
            return

        transaction_cost = self._transaction_cost(buy_value)
        # prezzo vero della transazione, comprensivo di costi
        total_cash_needed = buy_value + transaction_cost

        # se ho abbastanza liquidità, procedo
        if total_cash_needed > self.cash:
            buy_value = max(self.cash - transaction_cost, 0.0)
            total_cash_needed = buy_value + transaction_cost

        if buy_value < self.config.min_trade_value:
            return

        if total_cash_needed > self.cash:
            return

        # pagamento della liquidità per l'acquisto
        self.cash -= total_cash_needed

        # inizializzo il titolo se non esiste
        if symbol not in self.positions:
            self.positions[symbol] = {
                "value": 0.0,
                "cost_basis": 0.0
            }

        # aggiungo il valore di mercato della posizione
        self.positions[symbol]["value"] += buy_value

        # Il costo di transazione viene incluso nel costo fiscale.
        self.positions[symbol]["cost_basis"] += (
            buy_value + transaction_cost
        )

        self.trade_log.append({
            "date": date,
            "symbol": symbol,
            "side": "BUY",
            "trade_value": buy_value,
            "transaction_cost": transaction_cost,
            "basis_sold": 0.0,
            "realized_gain_after_costs": 0.0,
            "tax_paid": 0.0,
            "cash_after": self.cash,
            "portfolio_value_after": self.portfolio_value()
        })

    def rebalance(
        self,
        date: pd.Timestamp,
        target_weights: pd.Series
    ):
        """
        Ribilancia il portafoglio verso i pesi target.
        Viene richiamata il 1 Gennaio di ogni anno

        Procedura:
        1. normalizza i pesi target;
        2. vende le posizioni eccedenti o non più presenti;
        3. paga eventuali tasse sulle plusvalenze realizzate;
        4. ricalcola il capitale netto;
        5. compra le posizioni mancanti.
        """

        date = pd.Timestamp(date)
        # pulizia dati
        target_weights = target_weights.astype(float)
        target_weights = target_weights.replace(
            [np.inf, -np.inf],
            np.nan
        )
        target_weights = target_weights.dropna()
        target_weights = target_weights[target_weights > 0]

        if target_weights.empty:
            return

        target_weights = target_weights / target_weights.sum()

        # valore corrente
        current_total_value = self.portfolio_value()
        
        # calcolo il denaro che dovrei dare in base ai titoli
        initial_target_values = current_total_value * target_weights

        # 1. Vendo posizioni fuori target o sopra target.
        current_symbols = list(self.positions.keys())
        
        # itero su titoli che possiedo
        for symbol in current_symbols:
            current_value = self.positions[symbol]["value"]

            # calcolo il target di valore per il titolo
            target_value = float(
                initial_target_values.get(symbol, 0.0)
            )

            # valore corrente investito- quanto dovrei avere investito
            sale_value = max(current_value - target_value, 0.0)

            # se devo vendere, vendo
            if sale_value >= self.config.min_trade_value:
                self._sell(
                    date=date,
                    symbol=symbol,
                    sale_value=sale_value
                )

        # 2. Dopo vendite, costi e tasse, ricalcolo il valore netto.
        value_after_sells = self.portfolio_value()

        # devo ricalcolare i valori target sulla base del nuovo valore di portafoglio
        final_target_values = value_after_sells * target_weights

        buy_orders = {}

        for symbol, target_value in final_target_values.items():
            # prendo il valore corrente della posizione (0 se non esiste)
            current_value = self.positions.get(
                symbol,
                {"value": 0.0}
            )["value"]

            # vedo quanto dovrei comprare per raggiungere il target
            buy_value = max(float(target_value) - current_value, 0.0)

            if buy_value >= self.config.min_trade_value:
                buy_orders[symbol] = buy_value # segno quanto dovrei comprare per questo titolo

        if not buy_orders:
            return

        # calcolo il costo totale degli acquisti e n° di transazioni
        total_buy_value = sum(buy_orders.values())
        n_buys = len(buy_orders)

        # calcolo i costi totali (fissi + percentuali) degli acquisti
        fixed_costs = (
            self.config.transaction_fixed_cost * n_buys
        )

        proportional_costs = (
            self.config.transaction_rate * total_buy_value
        )

        # soldi necessari per eseguire tutti gli acquisti
        total_required_cash = (
            total_buy_value + fixed_costs + proportional_costs
        )

        # Se la liquidità non basta, riduco proporzionalmente gli acquisti.
        if total_required_cash > self.cash:
            # tolgo i costi fissi prima di tutto
            available_for_buy_values = self.cash - fixed_costs

            if available_for_buy_values <= 0:
                return

            # calcolo fattoredi scala per ridurre gli acquisti in modo proporzionale
            scale = available_for_buy_values / (
                total_buy_value * (1 + self.config.transaction_rate)
            )

            # controlli di sicurezza
            scale = max(min(scale, 1.0), 0.0)

            buy_orders = {
                symbol: value * scale # moltiplico ogni ordine per il fattore di scala
                for symbol, value in buy_orders.items()
            }

        # compro
        for symbol, buy_value in buy_orders.items():
            self._buy(
                date=date,
                symbol=symbol,
                buy_value=buy_value
            )

    def run(
        self,
        target_weights_by_date: Mapping[pd.Timestamp, pd.Series],
        start_date=None,
        end_date=None,
        strategy_name: str = ""
    ) -> dict:
        """
        Esegue il backtest.

        PARAMETRI
    
        target_weights_by_date : dict
            Dizionario {data_ribilanciamento: pesi_target}.
        start_date : str or Timestamp
            Data iniziale.
        end_date : str or Timestamp
            Data finale.
        strategy_name : str
            Nome della strategia.

        RITORNA
        
        dict con:
        - daily: andamento giornaliero;
        - trades: log degli ordini;
        - taxes: log fiscale.
        """

        # prima di iniziare una nuova simulazione, resetto lo stato del backtest
        self.reset()

        # converto le date dei ribilanciamenti in Timestamp e creo un dizionario
        schedule = {
            pd.Timestamp(date): weights
            for date, weights in target_weights_by_date.items()
        }

        # controlli di sicurezza sulle date di inizio e fine
        if start_date is None:
            if len(schedule) > 0:
                start_date = min(schedule.keys())
            else:
                start_date = self.returns.index.min()

        if end_date is None:
            end_date = self.returns.index.max()

        start_date = pd.Timestamp(start_date)
        end_date = pd.Timestamp(end_date)

        # estraggo tutte ledate di simulazione comprese fra start_date e end_date   
        dates = self.returns.loc[
            (self.returns.index >= start_date) &
            (self.returns.index <= end_date)
        ].index

        for date in dates:
            if date in schedule:
                self.rebalance(
                    date=date,
                    target_weights=schedule[date]
                )

            # applico i rendimenti giornalieri alle posizioni aperte
            self._apply_daily_returns(date)

            # aggiorno valore portfolio e loggo i dati giornalieri
            total_value = self.portfolio_value()

            self.daily_log.append({
                "date": date,
                "strategy": strategy_name,
                "portfolio_value": total_value,
                "cash": self.cash,
                "positions_value": total_value - self.cash,
                "loss_carryforward": self.loss_carryforward,
                "n_positions": len(self.positions)
            })

        # creo dataframe con dati giornalieri
        daily = pd.DataFrame(self.daily_log)

        if not daily.empty:
            daily = daily.set_index("date")

            # calcolo la variazione percentuale rispetto al giorno precedente
            daily["portfolio_return"] = (
                daily["portfolio_value"].pct_change()
            )

            # gestisco il caso di ritorni mancanti nel giorno 1
            # inserisco come valore il rapporto fra il valore del portafoglio e il capitale iniziale - 1
            first_return = (
                daily["portfolio_value"].iloc[0]
                / self.config.initial_capital
                - 1
            )

            # sostituisco i valori NaN con il rendimento calcolato sopra
            daily["portfolio_return"] = (
                daily["portfolio_return"].fillna(first_return)
            )

        trades = pd.DataFrame(self.trade_log)
        taxes = pd.DataFrame(self.tax_log)

        return {
            "daily": daily,
            "trades": trades,
            "taxes": taxes
        }