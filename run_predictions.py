import asyncio
import argparse
import ccxt.pro as ccxtpro

from src.util.data_structures.CappedListAsync import AsyncCappedList
from src.models.predict import load_prediction_model, fetch_real_time_crypto_data, real_time_predictions
from src.models.models_config import MLP_MODEL_CHECKPOINT_PATH

async def main(model_path: str, 
               crypto_pair_sym: str, 
               exchange_name: str,
               timeframe: str,
               device: str):
    '''
    The main function for running real-time predictions.
    
    Parameters:
        model_path (str): Path to the trained PyTorch model.
        crypto_pair_sym (str): Symbol of the cryptocurrency pair (e.g., BTC-USDT).
        exchange_name (str): The name of the exchange to get real-time data from.
        timeframe (str): Timeframe for the candlestick data.
        device (str): The device to perform inference on ('cpu' or 'cuda').
    '''

    historical_candle_data = AsyncCappedList(max_len=300) # Holds past OHLCV data
    candle_queue = asyncio.Queue(maxsize=100) # Holds the latest timeframe OHLCV data
    exchange_objs = {
        'okx': ccxtpro.okx({'enableRateLimit': True})
    }

    # Load model from model_path
    model = load_prediction_model(checkpoint_path=model_path, device=device)

    # Create exchange object to get real-time data from.
    exchange_obj = exchange_objs[exchange_name]

    fetch_task = asyncio.create_task(fetch_real_time_crypto_data(crypto_pair_sym=crypto_pair_sym,
                                                                 exchange_obj=exchange_obj,
                                                                 timeframe=timeframe,
                                                                 candle_queue=candle_queue,
                                                                 historical_data=historical_candle_data))
    prediction_task = asyncio.create_task(real_time_predictions(model=model, 
                                                                candle_queue=candle_queue, 
                                                                device=device))

    await asyncio.gather(fetch_task, prediction_task)

if __name__ == "__main__":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Real-time Crypto Trading Strategy with Model Prediction")

    # Parse arguments
    parser.add_argument("--model_path", default=MLP_MODEL_CHECKPOINT_PATH, type=str, help='Path to the PyTorch model (.pt or .ckpt file)')
    parser.add_argument("--crypto_pair_sym", default="BTC-USDT", type=str, help='Cryptocurrency pair symbol (e.g., BTC-USDT)')
    parser.add_argument("--exchange", default="okx", type=str, help='The name of the exchange to get data from (e.g., okx)')
    parser.add_argument("--timeframe", default="15m", type=str, help='The timeframe for candle data. (e.g., 15m)')
    parser.add_argument("--device", default="cpu", type=str, help='Device to perform inference on (e.g., cpu, cuda:0)')

    # Parse args
    args = parser.parse_args()

    # Run real-time prediction functions using args
    asyncio.run(main(model_path=args.model_path,
                     crypto_pair_sym=args.crypto_pair_sym,
                     exchange_name=args.exchange,
                     timeframe=args.timeframe,
                     device=args.device,))