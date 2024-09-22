import pandas as pd

from src.models.nn_algo_paper_model import NN_Algo
from src.models.train import train_model
from src.data_processing.Extract_Crypto_Data import collect_crypto_exchange_data
from src.data_processing.FeatureExtraction import extract_features_OHLCV, remove_columns_processed_data
from src.data_processing.Label import label_crypto_AB
from src.data_processing.config import COINBASE_15MIN_PATH, OKX_15MIN_PATH, FEATURES_EXCLUDED

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

    if fetch:
        # Fetch crypto from exchange
        data = collect_crypto_exchange_data(trading_sym, exchange_name=exchange, since=since, until=until, save_path=data_path + ".csv")
    else:
        data = pd.read_csv(data_path + ".csv")

    # Extract Features
    if process:
        processed_data = extract_features_OHLCV(data)
        processed_data.to_csv(data_path + "_processed.csv", index=False)
    
    # Remove unnecessary columns
    if remove_cols:
        save_path = data_path + "_fully_processed.csv"
        processed_data = remove_columns_processed_data(processed_data= processed_data, remove_columns=FEATURES_EXCLUDED, save_path=save_path)
    else:
        processed_data = pd.read_csv(data_path + "_fully_processed.csv")

    # Label data
    labeled_data = label_crypto_AB(data=processed_data)
    labeled_data.to_csv(data_path + "_labeled.csv", index=False)
    
    path = data_path + "_labeled.csv"
    model = NN_Algo()
    train_model(model, path)
    
main()


