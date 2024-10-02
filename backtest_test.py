import pandas as pd

from src.trading.backtests import backtest
from src.data_processing.config import BACKTEST_DATA_PATH
from src.models.models_config import MLP_MODEL_CHECKPOINT_PATH
from src.models.predict import load_prediction_model

data = pd.read_csv(BACKTEST_DATA_PATH)
input_size = 200
output_size = 3
model = load_prediction_model(MLP_MODEL_CHECKPOINT_PATH, 
                                input_size=input_size,
                                output_size=output_size,
                                device='cpu')
backtest(model, data)