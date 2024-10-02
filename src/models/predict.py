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
from ..data_processing.DataPipelines import prediction_pipeline_OHLCV

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
    data_tensor = torch.tensor(data.values, dtype=torch.float32).unsqueeze(0).to(device)
    
    with torch.no_grad():
        prediction = model(data_tensor)
        label = torch.argmax(prediction, dim=1)

    return int(label.item())

async def real_time_predictions(model, 
                                candle_queue: asyncio.Queue,
                                historical_data: AsyncCappedList, 
                                device: str = 'cpu',
                                backward_window: int = 5,) -> None:
    '''
    Performs real time predictions with the provided model.
    
    Parameters:
        model: The model to use for predictions on real time data.
        candle_queue (asyncio.Queue): An asynchronous queue to signal when new data is available for prediction.
        historical_data (AsyncCappedList): An asynchronous capped list to store historical OHLCV data.
        device (str): The device to perform inference on. (default = 'cpu')
        backward_window (int): The number of previous candles to use in predicting the next move. (default = 5)
    
    Returns:
        None
    '''
    while True:
        # Wait for the signal indicating new data is available
        await candle_queue.get()

        # Ensure enough historical data is available for the backward window
        if await historical_data.get_size() >= backward_window:
            # Get the most recent `backward_window` candles
            recent_data = await historical_data.get_list()

            # Convert to DataFrame for processing
            recent_data_df = pd.DataFrame(recent_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

            # Scale the features using the pre-fitted scaler
            features_scaled = prediction_pipeline_OHLCV(recent_data_df)
            
            # Select the last `backward_window` for predictions
            backward_data = features_scaled.iloc[-backward_window:]

            # Flatten the scaled features to create the input vector
            feature_vector = backward_data.values().flatten()
            
            # Make a prediction using the model
            prediction = predict(model=model, data=feature_vector, device=device)

            # Handle the prediction (e.g., place it in a result queue or take some action)
            print(f"Prediction: {prediction}")
    
async def fetch_real_time_crypto_data(crypto_pair_sym: str, 
                                    exchange_obj, 
                                    timeframe: str, 
                                    candle_queue: asyncio.Queue, 
                                    historical_data: AsyncCappedList,
                                    limit: int = 300) -> None:
    '''
    Fetch real-time cryptocurrency candle data and maintain historical data for prediction.

    Parameters:
        crypto_pair_sym (str): The symbol of the cryptocurrency pair to fetch data for (e.g., 'BTC/USDT').
        exchange_obj: The exchange object that provides methods to fetch real-time and historical OHLCV data.
        timeframe (str): The timeframe for each candle (e.g., '1m', '15m', '1h').
        candle_queue (asyncio.Queue): An asynchronous queue to signal when new data is available for prediction.
        historical_data (AsyncCappedList): An asynchronous capped list to store historical OHLCV data.
        limit (int, optional): The number of historical candles to fetch initially. Default is 300.

    Returns:
        Nothing
    '''
    
    sleep_duration_ms = _get_sleep_duration_ms(timeframe=timeframe) # Time to sleep between requests

    async with aiohttp.ClientSession() as session:
        while True:

            # Fetch real-time candle data for the timeframe
            candles = (await exchange_obj.watchOHLCV(symbol=crypto_pair_sym, timeframe=timeframe, limit=limit))[0]

            # Fetch Historical candle data
            if await historical_data.get_size() == 0:

                # Fetch initial limit candles for history
                historical_candles = await exchange_obj.fetchOHLCV(symbol=crypto_pair_sym, timeframe=timeframe, limit=limit)

                # Check if duplicate data exists
                if historical_candles[-1][0] == candles[0]:
                    del historical_candles[-1]
                
                # Add info to historical data
                await historical_data.extend(historical_candles)

            # Handle past and current candle data
            if await historical_data.get_size() > 0:
                # Add real-time candle to historical data
                await historical_data.append(candles)

            # Grab candle data
            historical_candles = await historical_data.get_list()

            # Combine candle info into a pandas DataFrame
            all_candles = historical_candles
            candle_data = pd.DataFrame(all_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # Put the candle data in the queue for prediction
            await candle_queue.put(1) # signal new data is available

            # Sleep until next request time
            start_time_ms = int(time.time() * 1000) # Current time in milliseconds
            ms_til_next_request = sleep_duration_ms - (start_time_ms % sleep_duration_ms) # Time in ms until the next request
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