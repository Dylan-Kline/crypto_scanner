import pandas as pd
import lightning as L
import websockets
import torch
import time
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

candle_queue = asyncio.Queue() # holds the latest candle data
processed_timestamps = set()

async def real_time_predictions(model, api_params: dict, device: str = 'cuda:0') -> int:
    '''
    Performs real time predictions with the provided model.
    
    Parameters:
        model: The model to use for predictions on real time data.
        api_params (dict): Holds the necessary information to decide what kind of candle
        data you want to use for predictions.
            Example columns: crypto_pair_sym: str, websocket_url: str, timeframe: str
        device(str): The device to perform inference on.
    
    Returns:
        Int: The predicted class label.
    '''
    
    while True:
        
        # Fetch real time candlestick data
        raw_data = await candle_queue.get()
        print(raw_data)
    
async def fetch_real_time_crypto_data(crypto_pair_sym: str, websocket_url: str, 
                                timeframe: str):
    candle_timeframe = "candle" + timeframe
    async with websockets.connect(websocket_url) as wb:
        
        # Subscribe to real-time candlestick data for the symbol
        params = {
            "op": "subscribe",
            "args": [{
                "channel": candle_timeframe,
                "instId": crypto_pair_sym
            }]
        }
        await wb.send(json.dumps(params))
        
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