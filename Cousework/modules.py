import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats import pearsonr
import yfinance as yf
import os
import matplotlib.pyplot as plt

import os
import pandas as pd
import yfinance as yf

class FileManager:
    """Handles file-related tasks like creating directories and saving/loading files."""
    @staticmethod
    def create_directory(path: str):
        """Creates directory if it doesn't already exist."""
        os.makedirs(path, exist_ok=True)
    
    @staticmethod
    def save_to_csv(data: pd.DataFrame, path: str):
        """Saves DataFrame to CSV."""
        data.to_csv(path)
    
    @staticmethod
    def load_from_csv(path: str) -> pd.DataFrame:
        """Loads DataFrame from CSV, raises FileNotFoundError if not found."""
        return pd.read_csv(path, index_col=0)


class Ticker:
    """Handles downloading and loading ticker data."""
    
    def __init__(self, symbol: str, start: str, end: str, interval: str):
        self.symbol = symbol
        self.start = start
        self.end = end
        self.interval = interval
        self.data = self._get_ticker_data()
    
    def download_data(self) -> pd.DataFrame:
        """Downloads ticker data using yfinance and saves to CSV."""
        data = yf.download(self.symbol, start=self.start, end=self.end, interval=self.interval)
        data.columns = data.columns.get_level_values(0)
        
        file_path = self._get_data_file_path()
        FileManager.save_to_csv(data, file_path)
        print(f'Data for {self.symbol} from {self.start} to {self.end} with interval {self.interval} saved \n')
        
        return data
    
    def load_data(self) -> pd.DataFrame:
        """Loads ticker data from CSV or downloads if not found."""
        try:
            data = FileManager.load_from_csv(self._get_data_file_path())
            return data
        except FileNotFoundError:
            print(f'Data for {self.symbol} from {self.start} to {self.end} with interval {self.interval} does not exist \n')
            print('Downloading data... \n')
            return self.download_data()
    
    def _get_ticker_data(self) -> pd.DataFrame:
        """Ensures data exists, downloading it if necessary."""
        FileManager.create_directory('data')
        return self.load_data()
    
    def _get_data_file_path(self) -> str:
        """Generates file path for saving/loading data."""
        return f'data/{self.symbol}_{self.interval}_{self.start}_{self.end}.csv'

    def __str__(self):
        return f'{self.symbol} from {self.start} to {self.end} with interval {self.interval} \n'


class Indicators:
    """Adds various technical indicators to ticker data."""
    
    def __init__(self, data: pd.DataFrame):
        self.data = data
    
    def add_ma(self, N: int):
        """Adds a Moving Average (MA) with a window of N."""
        self.data[f'MA{N}'] = self.data['price'].rolling(window=N).mean()
    
    def add_mstd(self, N: int):
        """Adds a Moving Standard Deviation (MSTD) with a window of N."""
        self.data[f'MSTD{N}'] = self.data['price'].rolling(window=N).std()
    
    def add_ema(self, mu: float):
        """Adds an Exponential Moving Average (EMA) with a span of mu."""
        self.data[f'EMA{mu}'] = self.data['price'].ewm(span=mu, adjust=False).mean()
    
    def add_ema_custom(self, mu: float, col:str):
        """Adds an Exponential Moving Average (EMA) with a span of mu."""
        self.data[f'EMA{mu}'] = self.data[col].ewm(span=mu, adjust=False).mean()

    def add_all_indicators(self):
        """Adds a set of default indicators."""
        self.add_ma(20)
        self.add_mstd(20)
        self.add_ema(20)


class PrepareData:
    """Prepares data for analysis, including splitting into training/testing sets."""
    
    def __init__(self, ticker: Ticker, splitting_point: str):
        self.ticker = ticker
        self.indicators = Indicators(self.ticker.data)  # Update ticker data with indicators
        self.splitting_point = splitting_point
    
    def train_test_split(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Splits data into training and testing sets."""
        train = self.ticker.data.loc[self.ticker.start:self.splitting_point]
        test = self.ticker.data.loc[self.splitting_point:self.ticker.end]
        return train, test
    
    def apply_indicators(self):
        """Applies a default set of indicators to the data."""
        self.indicators.add_all_indicators()


class SmaCrossoverStrategy:
    def __init__(self, data: pd.DataFrame, short_window: int = 20, long_window: int = 50):
        self.data = data.copy()
        self.short_window = short_window
        self.long_window = long_window
        
    def generate_signals(self):
        # Calculate the short and long SMAs
        self.data['MA_short'] = self.data['Close'].rolling(window=self.short_window).mean()
        self.data['MA_long'] = self.data['Close'].rolling(window=self.long_window).mean()
        
        # Create a signal: 1 when MA_short > MA_long, -1 otherwise.
        self.data['Signal'] = 0
        self.data.loc[self.data['MA_short'] > self.data['MA_long'], 'Signal'] = 1
        self.data.loc[self.data['MA_short'] <= self.data['MA_long'], 'Signal'] = -1
        
        # Shift signal to use as position (avoid lookahead bias)
        self.data['Position'] = self.data['Signal'].shift(1)
        return self.data
    

class BollingerBandsStrategy:
    def __init__(self, data: pd.DataFrame, window: int = 20, num_std: float = 2):
        self.data = data.copy()
        self.window = window
        self.num_std = num_std
        
    def generate_signals(self):
        # Calculate the moving average and standard deviation
        self.data['MA'] = self.data['Close'].rolling(window=self.window).mean()
        self.data['MSTD'] = self.data['Close'].rolling(window=self.window).std()
        
        # Compute upper and lower bands
        self.data['UpperBand'] = self.data['MA'] + self.num_std * self.data['MSTD']
        self.data['LowerBand'] = self.data['MA'] - self.num_std * self.data['MSTD']
        
        # Generate signals: buy when price < lower band, sell when price > upper band.
        self.data['Signal'] = 0
        self.data.loc[self.data['Close'] < self.data['LowerBand'], 'Signal'] = 1   # Buy signal
        self.data.loc[self.data['Close'] > self.data['UpperBand'], 'Signal'] = -1  # Sell signal
        
        # Create positions with a lag to avoid lookahead bias
        self.data['Position'] = self.data['Signal'].shift(1)
        return self.data


class MACDstrategy:
    def __init__(self, data: pd.DataFrame):
        self.data = data.copy()
    
    def generate_signals(self):

        ema12 = self.data['Close'].ewm(span=12, adjust=False).mean() 
        ema26 = self.data['Close'].ewm(span=26, adjust=False).mean()

        self.data['MACD'] = ema12 - ema26

        self.data['SignalLine'] = self.data['MACD'].ewm(span=9, adjust=False).mean()

        self.data['Signal'] = 0
        self.data.loc[self.data['MACD'] > self.data['SignalLine'], 'Signal'] = 1
        self.data.loc[self.data['MACD'] < self.data['SignalLine'], 'Signal'] = -1
        
        # Shift the signal to use the previous period's signal for execution
        self.data['Position'] = self.data['Signal'].shift(1)
        return self.data


class EmaTrendStrategy:
    def __init__(self, data: pd.DataFrame, ema_span: int = 20):
        self.data = data.copy()
        self.ema_span = ema_span
        
    def generate_signals(self):
        # Calculate the EMA of the closing prices
        self.data['EMA'] = self.data['Close'].ewm(span=self.ema_span, adjust=False).mean()
        
        # Generate signals: buy if Close > EMA, sell if Close < EMA.
        self.data['Signal'] = 0
        self.data.loc[self.data['Close'] > self.data['EMA'], 'Signal'] = 1
        self.data.loc[self.data['Close'] < self.data['EMA'], 'Signal'] = -1
        
        # Create positions by shifting the signal (to avoid lookahead bias)
        self.data['Position'] = self.data['Signal'].shift(1)
        return self.data


class MultiFactorStrategy:
    def __init__(self, data: pd.DataFrame,
                 ema_span: int = 50,
                 macd_short: int = 12,
                 macd_long: int = 26,
                 macd_signal_span: int = 9,
                 bb_window: int = 20,
                 bb_num_std: float = 2):
        self.data = data.copy()
        self.ema_span = ema_span
        self.macd_short = macd_short
        self.macd_long = macd_long
        self.macd_signal_span = macd_signal_span
        self.bb_window = bb_window
        self.bb_num_std = bb_num_std

    def generate_signals(self):
        # --- Trend Signal via EMA ---
        self.data['EMA_trend'] = self.data['Close'].ewm(span=self.ema_span, adjust=False).mean()
        # Bullish if Close is above EMA, bearish if below.
        self.data['TrendSignal'] = np.where(self.data['Close'] > self.data['EMA_trend'], 1, -1)
        
        # --- Momentum Signal via MACD ---
        ema_short = self.data['Close'].ewm(span=self.macd_short, adjust=False).mean()
        ema_long = self.data['Close'].ewm(span=self.macd_long, adjust=False).mean()
        self.data['MACD'] = ema_short - ema_long
        self.data['SignalLine'] = self.data['MACD'].ewm(span=self.macd_signal_span, adjust=False).mean()
        # Bullish if MACD is above its signal line, bearish otherwise.
        self.data['MACDSignal'] = np.where(self.data['MACD'] > self.data['SignalLine'], 1, -1)
        
        # --- Volatility Signal via Bollinger Bands ---
        self.data['BB_MA'] = self.data['Close'].rolling(window=self.bb_window).mean()
        self.data['BB_STD'] = self.data['Close'].rolling(window=self.bb_window).std()
        self.data['UpperBand'] = self.data['BB_MA'] + self.bb_num_std * self.data['BB_STD']
        self.data['LowerBand'] = self.data['BB_MA'] - self.bb_num_std * self.data['BB_STD']
        # Signal is +1 if price is below the lower band (potential oversold), -1 if above the upper band (potential overbought).
        self.data['BollingerSignal'] = 0
        self.data.loc[self.data['Close'] < self.data['LowerBand'], 'BollingerSignal'] = 1
        self.data.loc[self.data['Close'] > self.data['UpperBand'], 'BollingerSignal'] = -1
        
        # --- Composite Signal ---
        # Sum the individual signals. A higher composite score indicates stronger conviction.
        self.data['CompositeScore'] = (self.data['TrendSignal'] +
                                       self.data['MACDSignal'] +
                                       self.data['BollingerSignal'])
        
        # Define overall signal:
        # 1 (long) if composite score >= 2, -1 (short) if composite score <= -2, else neutral.
        self.data['Signal'] = 0
        self.data.loc[self.data['CompositeScore'] >= 2, 'Signal'] = 1
        self.data.loc[self.data['CompositeScore'] <= -2, 'Signal'] = -1
        
        # --- Position (Avoid Lookahead Bias) ---
        # Shift the signal so that today's signal is applied in the next period.
        self.data['Position'] = self.data['Signal'].shift(1)
        
        return self.data


def evaluate_performance(data: pd.DataFrame) -> dict:
    data = data.copy()
    
    data['Return'] = data['Close'].pct_change()
    
    data['StrategyReturn'] = data['Return'] * data['Position']
    
    data['CumulativeStrategyReturn'] = (1 + data['StrategyReturn']).cumprod() - 1
    
    sharpe_ratio = np.sqrt(252) * (data['StrategyReturn'].mean() / data['StrategyReturn'].std())
    cumulative_pnl = (1 + data['Close'].pct_change() * data['Position'].shift(1)).cumprod() -1  # assume initial capital=1

    cumulative = (1 + data['StrategyReturn']).cumprod()
    running_max = cumulative.cummax()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()
    
    return {
        "Sharpe Ratio": sharpe_ratio,
        "Max Drawdown": max_drawdown,
        "Cumulative Return": data['CumulativeStrategyReturn'].iloc[-1],
        # "Cumulative_pnl": cumulative_pnl
    }



# Usage Example
if __name__ == '__main__':    

    # Create a Ticker object
    Start = '2012-01-01'
    End = '2025-01-31'
    Interval= "1d" 
    ticker = Ticker(symbol='ZT=F', start=Start, end=End, interval='1d')

    # Prepare data
    splitting_point = '2023-08-01'
    prepare_data = PrepareData(ticker, splitting_point)
    train_data, test_data = prepare_data.train_test_split()


    data = ticker.data

    # # --- SMA Crossover Strategy Backtest ---
    sma_strategy = SmaCrossoverStrategy(data, short_window=20, long_window=50)
    sma_data = sma_strategy.generate_signals()
    sma_performance = evaluate_performance(sma_data)
    print("SMA Crossover Performance Metrics:")
    print(sma_performance)


    # # --- Bollinger Bands Strategy Backtest ---
    bollinger_strategy = BollingerBandsStrategy(data, window=20, num_std=2)
    bollinger_data = bollinger_strategy.generate_signals()
    bollinger_performance = evaluate_performance(bollinger_data)
    print("Bollinger Bands Performance Metrics:")
    print(bollinger_performance)

    # # --- EMA Trend Strategy Backtest ---
    ema_strategy = EmaTrendStrategy(data, ema_span=20)
    ema_data = ema_strategy.generate_signals()
    ema_performance = evaluate_performance(ema_data)
    print("EMA Trend Performance Metrics:")
    print(ema_performance)


    # --- MACD Trend Strategy Backtest ---

    macd_strategy = MACDstrategy(data)
    macd_data = macd_strategy.generate_signals()
    macd_performance = evaluate_performance(macd_data)
    print('MACD Performance Metrics:\n ', macd_performance)

    # --- Multi-Strategy ---

    multiStrat = MultiFactorStrategy(data)
    multiStrat_data = multiStrat.generate_signals()
    multiStrat_performance = evaluate_performance(multiStrat_data)
    print('Multi Strat Performance \n', multiStrat_performance)


    # --- 