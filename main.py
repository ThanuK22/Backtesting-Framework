import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List, Dict

class MovingAverageCrossStrategy:
    """
    Simple SMA crossover strategy:
      +1 = go long,
       0 = hold/flat,
      -1 = exit long (go flat).
    """
    def __init__(self, short_window: int = 40, long_window: int = 100):
        self.short_window = short_window
        self.long_window  = long_window

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        signals = pd.Series(0, index=data.index)
        short_sma = data['Close'].rolling(window=self.short_window, min_periods=1).mean()
        long_sma  = data['Close'].rolling(window=self.long_window,  min_periods=1).mean()
        signals[short_sma > long_sma] = 1
        signals[short_sma < long_sma] = -1
        # Only act on changes: +1 = enter, -1 = exit
        return signals.diff().fillna(0).astype(int)

class Backtest:
    """
    Backtesting engine with:
      - Any signal-based strategy
      - Commission & slippage modeling
      - Fixed-fraction position sizing
      - Optional stop-loss & take-profit
      - Performance metrics: CAGR, Sharpe, Sortino, Calmar, Max Drawdown
      - Trade log
      - Plots: price+signals, equity curve, drawdown
    """
    def __init__(
        self,
        data: pd.DataFrame,
        strategy: MovingAverageCrossStrategy,
        initial_capital: float = 100_000,
        commission: float = 0.002,
        slippage: float = 0.0005,
        percent_per_trade: float = 1.0,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None
    ):
        self.data = data.copy()
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.percent = percent_per_trade
        self.stop_loss = stop_loss
        self.take_profit = take_profit

    def run(self):
        signals = self.strategy.generate_signals(self.data)

        # Initialize portfolio DataFrame
        port = pd.DataFrame(index=self.data.index)
        port['signal'] = signals
        port['positions'] = 0
        port['cash'] = float(self.initial_capital)
        port['holdings'] = 0.0
        port['total'] = float(self.initial_capital)

        position = 0
        cash = self.initial_capital
        entry_price = None
        trade_log: List[Dict] = []

        for date, sig in signals.items():
            price = self.data.at[date, 'Close']
            equity = cash + position * price

            # Entry signal
            if sig == 1 and position == 0:
                allocation = equity * self.percent
                qty = int(allocation // (price * (1 + self.commission) + self.slippage * price))
                if qty > 0:
                    cost = qty * price * (1 + self.commission) + qty * price * self.slippage
                    cash -= cost
                    position = qty
                    entry_price = price
                    trade_log.append({
                        'Entry Date': date,
                        'Entry Price': price,
                        'Qty': qty
                    })

            # Exit signal or stop-loss / take-profit
            exit_cond = (
                (sig == -1 and position > 0) or
                (position > 0 and self.stop_loss is not None and price <= entry_price * (1 - self.stop_loss)) or
                (position > 0 and self.take_profit is not None and price >= entry_price * (1 + self.take_profit))
            )
            if exit_cond and position > 0:
                proceeds = position * price * (1 - self.commission) - position * price * self.slippage
                cash += proceeds
                trade_log[-1].update({
                    'Exit Date': date,
                    'Exit Price': price,
                    'Return': (price - entry_price) / entry_price
                })
                position = 0
                entry_price = None

            holdings = position * price
            total = cash + holdings

            port.at[date, 'positions'] = position
            port.at[date, 'cash'] = cash
            port.at[date, 'holdings'] = holdings
            port.at[date, 'total'] = total

        # Compute performance metrics
        port['returns'] = port['total'].pct_change().fillna(0)
        days = len(port)
        final_eq = port['total'].iloc[-1]
        cagr = (final_eq / self.initial_capital)**(252 / days) - 1

        ann_vol = port['returns'].std() * np.sqrt(252)
        sharpe = (port['returns'].mean() * np.sqrt(252) / port['returns'].std()
                  if port['returns'].std() > 0 else np.nan)

        neg_returns = port['returns'][port['returns'] < 0]
        sortino = (port['returns'].mean() * np.sqrt(252) / neg_returns.std()
                   if len(neg_returns) > 0 and neg_returns.std() > 0 else np.nan)

        cummax = port['total'].cummax()
        drawdown = (port['total'] - cummax) / cummax
        max_dd = drawdown.min()
        calmar = (cagr / abs(max_dd)) if max_dd < 0 else np.nan

        stats = {
            'CAGR': cagr,
            'Annual Vol': ann_vol,
            'Sharpe': sharpe,
            'Sortino': sortino,
            'Max Drawdown': max_dd,
            'Calmar': calmar,
            'Total Trades': len(trade_log)
        }

        trades_df = pd.DataFrame(trade_log)

        # Plot results
        fig, axes = plt.subplots(3, 1, sharex=True, figsize=(12, 12))

        # Price + signals
        axes[0].plot(self.data.index, self.data['Close'], label='Close Price')
        buys = signals == 1
        sells = signals == -1
        axes[0].plot(self.data.index[buys], self.data['Close'][buys], '^', markersize=8, label='Buy')
        axes[0].plot(self.data.index[sells], self.data['Close'][sells], 'v', markersize=8, label='Sell')
        axes[0].set_ylabel('Price')
        axes[0].legend()

        # Equity curve
        axes[1].plot(port.index, port['total'], label='Equity Curve')
        axes[1].set_ylabel('Equity')
        axes[1].legend()

        # Drawdown
        axes[2].fill_between(port.index, drawdown, 0, color='gray')
        axes[2].set_ylabel('Drawdown')
        axes[2].set_xlabel('Date')

        plt.tight_layout()
        plt.show()

        return stats, port, trades_df

if __name__ == "__main__":
    # Load your complex OHLCV dataset
    df = pd.read_csv('data/more_trades_stock_data.csv', index_col=0, parse_dates=True)

    # Configure strategy & backtester
    strategy = MovingAverageCrossStrategy(short_window=10, long_window=30)
    backtester = Backtest(
        data=df,
        strategy=strategy,
        initial_capital=100_000,
        commission=0.001,
        slippage=0.0002,
        percent_per_trade=0.5,   # 50% of equity per trade
        stop_loss=0.05,          # 5% stop-loss
        take_profit=0.10         # 10% take-profit
    )

    stats, portfolio, trades = backtester.run()

    print("Performance Metrics:")
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    print("\nTrade Log:")
    print(trades)
