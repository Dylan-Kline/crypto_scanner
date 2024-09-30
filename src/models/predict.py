import pandas as pd
import lightning as L
import time
from datetime import datetime, timedelta
import torch
import ccxt.pro as ccxtpro
import asyncio
import json
import aiohttp

from lightning import LightningModule

from ..util.data_structures.CappedListAsync import AsyncCappedList
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

async def real_time_predictions(model, candle_queue: asyncio.Queue, device: str = 'cpu') -> int:
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
                                    historical_data: AsyncCappedList,
                                    limit: int = 300):
    
    sleep_duration_ms = _get_sleep_duration_ms(timeframe=timeframe) # Time to sleep between requests
    print(sleep_duration_ms)

    async with aiohttp.ClientSession() as session:
        while True:

            # Fetch real-time candle data for the timeframe
            candles = (await exchange_obj.watchOHLCV(symbol=crypto_pair_sym, timeframe=timeframe, limit=limit))[0]

            # Fetch Historical candle data
            if await historical_data.get_size() == 0:

                # Fetch initial limit candles for history
                historical_candles = await exchange_obj.fetchOHLCV(symbol=crypto_pair_sym, timeframe=timeframe, limit=limit)
                print(historical_candles[-3:])

                # Check if duplicate data exists
                if historical_candles[-1][0] == candles[0]:
                    del historical_candles[-1]
                
                # Add info to historical data
                await historical_data.extend(historical_candles)

            # Handle past and current candle data
            if await historical_data.get_size() > 0:

                # Retrieve last stored candled data
                last_stored = await historical_data.get_last()

                # Check for duplicates
                if last_stored and last_stored[0] == candles[0]:
                    print("Skipping duplicate entry.")
                else:
                    # Add real-time candle to historical data
                    await historical_data.append(candles)

            print("Updated data:", (await historical_data.get_list())[-3:])
            historical_candles = await historical_data.get_list()

            # Combine candle info into a pandas DataFrame
            all_candles = historical_candles
            candle_data = pd.DataFrame(all_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            print(candle_data)
            # Put the candle data in the queue for prediction
            await candle_queue.put(candle_data)

            start_time_ms = int(time.time() * 1000) # Current time in milliseconds
            ms_til_next_request = sleep_duration_ms - (start_time_ms % sleep_duration_ms) # Time in ms until the next request
        
            print(start_time_ms)
            print(start_time_ms % sleep_duration_ms)
            print(ms_til_next_request)
            
            # Sleep for the sleep duration
            await asyncio.sleep(ms_til_next_request // 1000)

def _get_sleep_duration_ms(timeframe: str) -> int:
    '''
    Calculates the number of ms to wait before requesting data again from the exchange.
    
    Parameters:
        timeframe (str): The string representation of the time to wait.
        
    Returns:
        int: number of ms to wait
    '''

    if timeframe.endswith('m'):
        ms = int(timeframe[:-1]) * 60000
    elif timeframe.endswith('h'):
        ms = int(timeframe[:-1]) * 3600000
    else:
        raise ValueError("Unsupported timeframe. Use 'm' for minutes or 'h' for hours.")

    return ms