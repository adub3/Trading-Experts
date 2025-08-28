import os
import pandas as pd
import yfinance as yf

RAW_DIR = os.path.join("data", "raw")
os.makedirs(RAW_DIR, exist_ok=True)

def get_data(symbol, start_date=None, end_date=None, source="yahoo"):
    df = load_from_cache(symbol)
    if df is None:
        df = download_data(symbol, source=source)
    if start_date:
        df = df[df.index >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df.index <= pd.to_datetime(end_date)]
    return df

def load_from_cache(symbol):
    filepath = os.path.join(RAW_DIR, f"{symbol}.csv")
    if os.path.exists(filepath):
        return pd.read_csv(filepath, index_col=0, parse_dates=True)
    return None

def download_data(symbol, source="yahoo", start="1990-01-01", end=None):
    if source == "yahoo":
        df = yf.download(symbol, start=start, end=end, progress=False)
        if df.empty:
            raise ValueError(f"No data found for {symbol} in range {start} to {end}")
        df = df[["Open", "High", "Low", "Close", "Volume"]]
    else:
        raise ValueError(f"Unsupported data source: {source}")
    filepath = os.path.join(RAW_DIR, f"{symbol}.csv")
    df.to_csv(filepath)
    return df

if __name__ == "__main__":
    df = get_data("SPY", start_date="2020-01-01", end_date="2021-01-01")
    print(df.head())
