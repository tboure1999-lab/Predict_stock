import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class StockDataHandler:
    def __init__(self):
        self.scaler = MinMaxScaler((0, 1))

    def get_stock_data(self, ticker, period='2y'):
        try:
            df = yf.Ticker(ticker).history(period=period)
            if df.empty:
                return None
            return df[['Open', 'High', 'Low', 'Close', 'Volume']]
        except:
            return None

    def calculate_technical_indicators(self, df):
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = -delta.where(delta < 0, 0).rolling(14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        df['MA_20'] = df['Close'].rolling(20).mean()
        df['MA_50'] = df['Close'].rolling(50).mean()

        exp1 = df['Close'].ewm(span=12).mean()
        exp2 = df['Close'].ewm(span=26).mean()
        df['MACD'] = exp1 - exp2
        df['Signal'] = df['MACD'].ewm(span=9).mean()

        df['Volatility'] = df['Close'].rolling(20).std()
        df['Price_Change'] = df['Close'].pct_change()

        return df.dropna()

    def get_latest_sequence(self, df, seq_len):
        features = [
            'Close', 'Volume', 'RSI', 'MA_20', 'MA_50',
            'MACD', 'Signal', 'Volatility', 'Price_Change'
        ]
        data = df[features].values
        scaled = self.scaler.fit_transform(data)
        return np.array([scaled[-seq_len:]])

    def get_current_price(self, ticker):
        try:
            df = yf.Ticker(ticker).history(period='1d')
            return round(df['Close'].iloc[-1], 2)
        except:
            return None

    def get_stock_info(self, ticker):
        stock = yf.Ticker(ticker)
        info = stock.info
        return {
            'name': info.get('longName', ticker),
            'sector': info.get('sector', 'N/A'),
            'current_price': self.get_current_price(ticker),
            'market_cap': info.get('marketCap', 'N/A')
        }
