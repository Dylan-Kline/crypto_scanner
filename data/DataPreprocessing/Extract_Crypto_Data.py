import ccxt
import pandas as pd
import numpy as np

from .config import COINBASE_API_SECRET, COINBASE_API_KEY, COINBASE_15MIN_PATH

def collect_crypto_exchange_data(crypto_pair_sym: str, 
                                 timeframe: str = '15m', limit: int = None, 
                                 since: int = None, until: int = None) -> pd.DataFrame:
    '''
        Collect OHLCV data from the coinbase cryptocurrency exchange.
        
        Parameters:
            crypto_pair_sym (str): The trading pair symbol (such as 'BTC-USD')
            timeframe (str): Time interval for candlesticks (such as '4h', '2d')
            limit (int): Number of data points to fetch in total (default = None).
            since (int): The UNIX timestamp of the start date to fetch from (default = None).
            until (int): The UNI timestamp of the end date to fetch data to (default = None)
            
        Returns:
            pd.DataFrame: A DataFrame containing OHLCV data
    '''
    
    exchange = ccxt.coinbase({
        'enableRateLimit': True
        })
    
    # Default timestamps from August 17, 2017 to December 4, 2022
    if not since or not until:
        since = int(pd.Timestamp(f'2017-08-17').timestamp() * 1000)
        until = int(pd.Timestamp(f'2022-12-04').timestamp() * 1000)
        
    all_ohlcv = list()
    limit = 350 # Max num of data points per request
    current_since = since
    
    while current_since < until:
        ohlcv = exchange.fetch_ohlcv(symbol=crypto_pair_sym, timeframe= timeframe, since = current_since, limit = limit)
        
        if len(ohlcv) == 0:
            print("No ohlcv data fetched.")
            break
        
        all_ohlcv.extend(ohlcv)
        current_since = ohlcv[-1][0] + 1
        
    ohlcv_df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    
    # Filter the data to ensure it's within the correct timespan
    ohlcv_df = ohlcv_df[(ohlcv_df['timestamp'] >= since) & (ohlcv_df['timestamp'] <= until)]
    ohlcv_df.to_csv(COINBASE_15MIN_PATH)
    return ohlcv_df

def fetch_coingecko_hlc(crypto: str, vs_currency: str, days: str, interval: str):
    '''
    Fetch OHLC data for a specified cryptocurrency from CoinGecko API.
    
    Parameters:
        crypto (str): The cryptocurrency symbol ('bitcoin', 'ethereum').
        vs_currency (str): Comparison currency to price the crypto ('USD', etc.).
        days (str): Number of days you want the data for ('1', '30', 'max').
        interval (str): Data interval ('1m', '5m', '1h', '1d').
        
    Returns:
        pd.DataFrame: A DataFrame containing OHLC data.
    '''
    pass
    
    