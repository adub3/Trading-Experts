import yfinance as yf
def get_data(ticker):
    data = yf.download(ticker, start='2020-01-01', end='2023-01-01')
    return data

get_data('AAPL')