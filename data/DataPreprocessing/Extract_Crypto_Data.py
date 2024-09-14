import ccxt
import pandas as pd
import numpy as np

from .config import COINBASE_API_SECRET, COINBASE_API_KEY, COINBASE_15MIN_PATH, OKX_15MIN_PATH

def collect_crypto_exchange_data(crypto_pair_sym: str, exchange_name: str,
                                 timeframe: str = '15m', limit: int = None, 
                                 since: int = None, until: int = None, save_path: str = None) -> pd.DataFrame:
    '''
        Collect OHLCV data from the coinbase cryptocurrency exchange.
        
        Parameters:
            crypto_pair_sym (str): The trading pair symbol (such as 'BTC-USD').
            exchange_name (str): The exchange to pull data from ('coinbase').
            timeframe (str): Time interval for candlesticks (such as '4h', '2d')
            limit (int): Number of data points to fetch in total (default = None).
            since (int): The UNIX timestamp of the start date to fetch from (default = None).
            until (int): The UNI timestamp of the end date to fetch data to (default = None)
            save_path (str): The file path to save the fetched OHLCV data
        Returns:
            pd.DataFrame: A DataFrame containing OHLCV data
    '''

    exchanges = {
        'coinbase':ccxt.coinbase({'enableRateLimit': True}),
        'okx': ccxt.okx({'enableRateLimit': True})
    }

    timeframe_paths = {
        'coinbase': {
            '15m': COINBASE_15MIN_PATH
        },
        'okx': {
            '15m': OKX_15MIN_PATH
        }
    }

    # Ensure the exchange name and timeframe are valid
    if exchange_name not in exchanges:
        raise ValueError(f"Exchange '{exchange_name}' is not supported.")
    if timeframe not in timeframe_paths[exchange_name]:
        raise ValueError(f"Timeframe '{timeframe}' is not supported for exchange '{exchange_name}'.")
    
    exchange = exchanges[exchange_name]

    # Default timestamps from August 17, 2017 to December 4, 2022
    if not since or not until:
        since = int(pd.Timestamp(f'2017-08-17').timestamp() * 1000)
        until = int(pd.Timestamp(f'2022-12-04').timestamp() * 1000)
        
    all_ohlcv = list()
    current_since = since
    
    while current_since < until:

        if limit:
            ohlcv = exchange.fetch_ohlcv(symbol=crypto_pair_sym, timeframe= timeframe, since = current_since, limit = limit)
        else:
            ohlcv = exchange.fetch_ohlcv(symbol=crypto_pair_sym, timeframe= timeframe, since = current_since)
        
        if len(ohlcv) == 0:
            print("No ohlcv data fetched.")
            break
        
        all_ohlcv.extend(ohlcv)
        current_since = ohlcv[-1][0] + 1
        
    ohlcv_df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    
    # Filter the data to ensure it's within the correct timespan and conver to readable timestamp
    if not ohlcv_df.empty:
        ohlcv_df = ohlcv_df[(ohlcv_df['timestamp'] >= since) & (ohlcv_df['timestamp'] <= until)]
        ohlcv_df['timestamp'] = pd.to_datetime(ohlcv_df['timestamp'], unit='ms', utc=True).dt.tz_convert('America/New_York').dt.strftime('%Y-%m-%d %H:%M:%S')
    else:
        print("Empty ohlcv_df")
    
    # Save data version for timespan
    since_date_utc = pd.to_datetime(since, unit='ms', utc=True)
    since_date_local = since_date_utc.tz_convert('America/New_York').strftime('%Y-%m-%d')
    until_date_utc = pd.to_datetime(until, unit='ms', utc=True)
    until_date_local = until_date_utc.tz_convert('America/New_York').strftime('%Y-%m-%d')

    if not save_path:
        save_path = timeframe_paths[exchange_name][timeframe] +  + f"_{since_date_local}_{until_date_local}.csv"
    ohlcv_df.to_csv(save_path, index = False)

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
    
    