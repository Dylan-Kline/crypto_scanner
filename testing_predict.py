import asyncio
import ccxt.pro as ccxtpro
from asyncio import Queue

from src.util.data_structures.CappedListAsync import AsyncCappedList
from src.models.predict import real_time_predictions, load_prediction_model, fetch_real_time_crypto_data 
from src.models.models_config import MLP_MODEL_CHECKPOINT_PATH

asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
async def main():
    
    # Init asyncio queue and async historical data
    candle_queue = Queue()
    historical_data = AsyncCappedList(max_len=300)
    
    api_params = {
        "crypto_pair_sym": 'BTC-USDT', 
        "timeframe": '15m'
    }
    exchange = ccxtpro.okx({'enableRateLimit': True})
    input_size = 200
    output_size = 3
    model = load_prediction_model(MLP_MODEL_CHECKPOINT_PATH, 
                                  input_size=input_size,
                                  output_size=output_size,
                                  device='cpu')
    
    try:
        subscriber_task = asyncio.create_task(fetch_real_time_crypto_data(crypto_pair_sym=api_params['crypto_pair_sym'],
                                                                        candle_queue=candle_queue,
                                                                        historical_data=historical_data,
                                                                        exchange_obj=exchange,
                                                                        timeframe=api_params['timeframe']))
        predictor_task = asyncio.create_task(real_time_predictions(model=model, 
                                                                candle_queue=candle_queue,
                                                                historical_data=historical_data, 
                                                                device='cpu'))
        
        # Wait for both tasks to complete
        await asyncio.gather(subscriber_task, predictor_task)
    finally:
        await exchange.close()
    
if __name__ == '__main__':
    asyncio.run(main())