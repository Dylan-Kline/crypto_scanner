import asyncio
import ccxt.pro as ccxtpro

from src.models.predict import real_time_predictions, load_prediction_model, fetch_real_time_crypto_data 
from src.models.models_config import MLP_MODEL_CHECKPOINT_PATH

asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
async def main():
    
    api_params = {
        "crypto_pair_sym": 'BTC-USDT', 
        "timeframe": '15m'
    }
    exchange = ccxtpro.okx({'enableRateLimit': True})
    model = load_prediction_model(MLP_MODEL_CHECKPOINT_PATH, device='cpu')
    subscriber_task = asyncio.create_task(fetch_real_time_crypto_data(crypto_pair_sym=api_params['crypto_pair_sym'],
                                                                      exchange_obj=exchange,
                                                                      timeframe=api_params['timeframe']))
    predictor_task = asyncio.create_task(real_time_predictions(model=model, api_params=api_params, device='cpu'))
    
    # Wait for both tasks to complete
    await asyncio.gather(subscriber_task, predictor_task)
    
if __name__ == '__main__':
    asyncio.run(main())