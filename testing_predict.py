import asyncio

from src.models.predict import real_time_predictions, load_prediction_model, fetch_real_time_crypto_data 
from src.models.models_config import MLP_MODEL_CHECKPOINT_PATH

async def main():
    
    api_params = {
        "crypto_pair_sym": 'BTC-USDT', 
        "websocket_url": 'wss://ws.okx.com:8443/ws/v5/public',
        "timeframe": '15m'
    }
    model = load_prediction_model(MLP_MODEL_CHECKPOINT_PATH)
    subscriber_task = asyncio.create_task(fetch_real_time_crypto_data(crypto_pair_sym=api_params['crypto_pair_sym'],
                                                                      websocket_url=api_params['websocket_url'],
                                                                      timeframe=api_params['timeframe']))
    predictor_task = asyncio.create_task(real_time_predictions(model=model, api_params=api_params))
    
    # Wait for both tasks to complete
    await asyncio.gather(subscriber_task, predictor_task)
    
if __name__ == '__main__':
    asyncio.run(main())