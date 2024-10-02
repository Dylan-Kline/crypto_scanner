import pandas as pd

from torch_lr_finder import LRFinder
from torch import nn
import torch.optim as optim

from src.models.nn_algo_paper_model import NN_Algo, NN_algo_model
from src.models.train import train_model, _load_data
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
    find_lr = False
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
    if process:
        training_pipeline_OHLCV(raw_data=data, candle_interval='15min', backward_window=bWin, forward_window=fWin)
    train_path = CRYPTO_15MIN_PATH + f"scaled_labeled_bWin_{bWin}_fWin_{fWin}_train.csv"
    val_path = CRYPTO_15MIN_PATH + f"scaled_labeled_bWin_{bWin}_fWin_{fWin}_val.csv"
    
    if find_lr:
        model = NN_algo_model(200, 3)
    else:
        model = NN_Algo(200, 3)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.0001)
    
    # find optimal learning rate
    if find_lr:
        train_loader, val_loader = _load_data(train_path, val_path)
        lr_finder = LRFinder(model, optimizer, criterion, device='cuda:0')
        lr_finder.range_test(train_loader, val_loader)
        lr_finder.plot()
        lr_finder.reset()
        
    else:
        train_model(model, train_path=train_path, val_path=val_path)
    
main()


