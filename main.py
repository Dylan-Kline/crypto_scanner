import pandas as pd

from src.models.nn_algo_paper_model import NN_Algo
from src.models.train import train_model
from src.data_processing.Extract_Crypto_Data import fetch_historical_crypto_data
from src.data_processing.FeatureExtraction import extract_features_OHLCV, remove_columns_processed_data
from src.data_processing.Label import label_crypto_AB
from src.data_processing.config import COINBASE_15MIN_PATH, OKX_15MIN_PATH, FEATURES_EXCLUDED, CRYPTO_15MIN_PATH
from src.data_processing.DataPipelines import training_pipeline_OHLCV

def main():

    # Crypto fetching parameters
    trading_sym = 'BTC-USDT'
    exchange = 'okx'
    since_date = '2018-01-17'
    until_date = '2022-12-04'
    data_path = OKX_15MIN_PATH + f"_{since_date}_{until_date}"
    since = int(pd.Timestamp(f"{since_date}", tz='America/New_York').timestamp() * 1000)
    until = int(pd.Timestamp(f"{until_date}", tz='America/New_York').timestamp() * 1000)
    fetch = False
    process = False
    remove_cols = False
    backtest_since_date = '2022-01-01'
    backtest_until_date = '2022-04-01'

    if fetch:
        # Fetch crypto from exchange
        data = fetch_historical_crypto_data(trading_sym, exchange_name=exchange, since=since, until=until, save_path=data_path + ".csv", save = True)
    else:
        data = pd.read_csv(data_path + ".csv")
    
    bWin = 5
    fWin = 1
    training_pipeline_OHLCV(raw_data=data, candle_interval='15min', backward_window=bWin, forward_window=fWin)
    train_path = CRYPTO_15MIN_PATH + f"scaled_labeled_bWin_{bWin}_fWin_{fWin}_train.csv"
    val_path = CRYPTO_15MIN_PATH + f"scaled_labeled_bWin_{bWin}_fWin_{fWin}_val.csv"
    model = NN_Algo(200, 5)
    train_model(model, train_path=train_path, val_path=val_path)
    
main()


