import pandas as pd

from data.DataPreprocessing.Extract_Crypto_Data import collect_crypto_exchange_data
from data.DataPreprocessing.FeatureExtraction import extract_features_OHLCV
from data.DataPreprocessing.config import COINBASE_15MIN_PATH

def main():
    trading_sym = 'BTC-USDT'
    data_path = "C:\projects\crypto_scanner\data\datasets\\15min\coinbase\coinbase_15min_2017-08-17_2022-12-04.csv"
    exchange = 'okx'
    since_date = '2024-09-14'
    until_date = '2024-09-15'
    since = int(pd.Timestamp(f"{since_date}", tz='America/New_York').timestamp() * 1000)
    until = int(pd.Timestamp(f"{until_date}", tz='America/New_York').timestamp() * 1000)

    data = collect_crypto_exchange_data(trading_sym, exchange_name=exchange, since=since, until=until)
    # data = pd.read_csv(data_path)
    print(data)
    data = extract_features_OHLCV(data)
    print(data[(data['three_black_crows'] < 0) | (data['three_black_crows'] > 0)])
    
main()


