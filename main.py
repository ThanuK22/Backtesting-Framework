import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class MovingAverageCrossStrategy:
    """
    SMA crossover:
      1 = buy signal, -1 = sell signal, 0 = no new signal.
    """
    def __init__(self, short_window=40, long_window=100):
        self.short_window = short_window
        self.long_window  = long_window

    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        # zero signal by default
        signals = pd.Series(0, index=data.index)

        # rolling SMAs
        short_sma = data['Close'].rolling(self.short_window, min_periods=1).mean()
        long_sma  = data['Close'].rolling(self.long_window,  min_periods=1).mean()

        #  1 when short > long, -1 when short < long
        signals[short_sma > long_sma] =  1
        signals[short_sma < long_sma] = -1

        # only act on changes
        return signals.diff().fillna(0).astype(int)

class Backtest:
    """
    Backtester with commission/slippage, metrics, trade‐count, and plots.
    """
    def __init__(self, data: pd.DataFrame, strategy: MovingAverageCrossStrategy,
                 initial_capital: float = 100_000,
                 commission: float    = 0.002,
                 slippage: float      = 0.0005):
        self.data            = data.copy()
        self.strategy        = strategy
        self.initial_capital = initial_capital
        self.commission      = commission
        self.slippage        = slippage

    def run(self):
        # generate entry/exit signals
        self.signals = self.strategy.generate_signals(self.data)

        # count total trades
        n_trades = int(self.signals.abs().sum())
        print(f"Total trades signalled: {n_trades}")
        if n_trades == 0:
            print("⚠️  Warning: no trades were generated. "
                  "Try smaller SMA windows or more data.")

        # init portfolio dataframe
        port = pd.DataFrame(index=self.data.index)
        port['signal']    = self.signals
        port['positions'] = 0
        port['cash']      = self.initial_capital
        port['holdings']  = 0.0
        port['total']     = self.initial_capital

        pos  = 0
        cash = self.initial_capital

        # simulate
        for date, sig in self.signals.items():
            price = self.data.at[date, 'Close']

            # ENTER long
            if sig == 1 and pos == 0:
                qty  = cash // (price * (1 + self.commission) + self.slippage * price)
                cost = qty * price * (1 + self.commission) + self.slippage * qty * price
                pos  = qty
                cash -= cost

            # EXIT long
            elif sig == -1 and pos > 0:
                proceeds = pos * price * (1 - self.commission) - self.slippage * pos * price
                cash    += proceeds
                pos      = 0

            holdings = pos * price
            total    = cash + holdings

            port.at[date, 'positions'] = pos
            port.at[date, 'cash']      = cash
            port.at[date, 'holdings']  = holdings
            port.at[date, 'total']     = total

        # performance
        port['returns'] = port['total'].pct_change().fillna(0)
        r_mean = port['returns'].mean()
        r_std  = port['returns'].std()

        if r_std > 0:
            sharpe = np.sqrt(252) * r_mean / r_std
        else:
            sharpe = np.nan

        cummax   = port['total'].cummax()
        drawdown = (port['total'] - cummax) / cummax
        max_dd   = drawdown.min()

        # plot
        fig, (ax1, ax2) = plt.subplots(2,1, sharex=True, figsize=(12,8))

        # price + markers
        ax1.plot(self.data.index, self.data['Close'], label='Close')
        buys  = self.signals == 1
        sells = self.signals == -1
        ax1.plot(self.data.index[buys],  self.data['Close'][buys],  '^', markersize=10, label='Buy')
        ax1.plot(self.data.index[sells], self.data['Close'][sells], 'v', markersize=10, label='Sell')
        ax1.set_ylabel('Price')
        ax1.legend()

        # equity curve
        ax2.plot(port.index, port['total'], label='Equity Curve')
        ax2.set_ylabel('Portfolio Value')
        ax2.set_xlabel('Date')
        ax2.legend()

        plt.tight_layout()
        plt.show()

        return {'Sharpe Ratio': sharpe, 'Max Drawdown': max_dd}, port

if __name__ == "__main__":
    # load your data
    df = pd.read_csv('data/stock_data.csv', index_col=0, parse_dates=True)

    # for a short sample, use smaller windows
    strat = MovingAverageCrossStrategy(short_window=3, long_window=5)

    backtester = Backtest(
        data=df,
        strategy=strat,
        initial_capital=100_000,
        commission=0.001,
        slippage=0.0002
    )

    stats, portfolio = backtester.run()
    print(f"Sharpe Ratio: {stats['Sharpe Ratio']}")
    print(f"Max Drawdown: {stats['Max Drawdown']:.2%}")
