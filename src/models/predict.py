import pandas as pd
import lightning as L
import time
import argparse
import torch
import ccxt.pro as ccxtpro
import asyncio
import json

from lightning import LightningModule

from .nn_algo_paper_model import NN_Algo
from ..data_processing.Extract_Crypto_Data import fetch_historical_crypto_data

# TODO create dynamic model loading functionality

def _create_model_from_hparams(hparams):
    '''
    Constructs the model from the provided hyperparameters.
    
    Parameters:
        hparams: Hyperparameters of model to create
        
    Returns:
        created model.
    '''
    
    if hparams['model_type'] == 'NN_Algo':
        pass
    
def _process_prediction(prediction):
    print(f"Prediction: {prediction.item()}")

def load_prediction_model(checkpoint_path: str, device: str = 'cuda:0'):
    '''
    Loads the lightning model saved at the checkpoint_path and loads it to the given device.
    
    Parameters:
        checkpoint_path (str): The file path containing the .ckpt model checkpoint.
        device (str): The device to load the model to ('cpu' or 'cuda:device id') (default = 'cuda:0')
        
    Returns:
        Loaded model in evaluation mode.
    '''
    
    model = NN_Algo.load_from_checkpoint(checkpoint_path=checkpoint_path, map_location=device)
    model.eval()
    return model
    
    # # Load the checkpoint
    # checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # if 'hyper_parameters' in checkpoint:
    #     hparams = checkpoint['hyper_paramters']
        
    #     # Reconstruct model based on hparams
    #     model = create

def predict(model, data: pd.DataFrame, device: str = 'cpu') -> int:
    '''
    Performs inference using the model on the provided data.
    
    Parameters:
        model: The classification algorithm to perform inference with.
        data: The processed data to perform inference on.
        device: The device to run inference on.
        
    Returns:
        class label (int): Returns 0 for 'buy', 1 for 'hold', and 2 for 'sell'
    '''
    model.eval()

    # Convert the data to a tensor
    data_tensor = torch.tensor(data.values, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        prediction = model(data_tensor)
        label = torch.argmax(prediction, dim=1)

    return int(label.item())

async def real_time_predictions(model, candle_queue: asyncio.Queue, device: str = 'cuda:0') -> int:
    '''
    Performs real time predictions with the provided model.
    
    Parameters:
        model: The model to use for predictions on real time data.
        device(str): The device to perform inference on.
    
    Returns:
        Int: The predicted class label.
    '''
    
    while True:
        
        # Fetch real time candlestick data
        raw_data = await candle_queue.get()
        print(raw_data)
    
async def fetch_real_time_crypto_data(crypto_pair_sym: str, 
                                    exchange_obj, 
                                    timeframe: str, 
                                    candle_queue: asyncio.Queue, 
                                    limit: int = 300):
    
    # Fetch initial 100 candles for history
    historical_candles = await exchange_obj.fetchOHLCV(symbol=crypto_pair_sym, timeframe=timeframe, limit=limit)
    
    candles = await exchange_obj.watchOHLCV(symbol=crypto_pair_sym, timeframe=timeframe, limit=limit)
    print(candles)
    
    # Handle incoming messages from websocket
    while True:
        message = await wb.recv()
        data = json.loads(message)
        print(data)
        
        # Make sure data contains candlestick data
        if 'data' in data and data['arg']['channel'] == candle_timeframe:
            print(data)
            candle = data['data'][0]
            timestamp = pd.to_datetime(candle[0], unit='ms')
            open_price = float(candle[1])
            high_price = float(candle[2])
            low_price = float(candle[3])
            close_price = float(candle[4])
            volume = float(candle[5])
            
            # Create a candle dictionary
            candle_data = {
                'timestamp': timestamp,
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': volume
            }
            
            candle_data = pd.DataFrame(candle_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

            # Put the candle data in the queue for prediction
            await candle_queue.put(candle_data)
