import os
from dotenv import load_dotenv

load_dotenv()
current_dir = os.getcwd()

FEATURES_EXCLUDED = ['ema_1', 'ema_20', 'ema_50', 'ema_100', 'timestamp']
FEATURE_SCALER_PATH = os.path.join(current_dir, "src\\data_processing\\scalers\\")

COINBASE_API_KEY = os.getenv("COINBASE_API_KEY")
COINBASE_API_SECRET = os.getenv("COINBASE_API_SECRET")

CRYPTO_15MIN_PATH = os.path.join(current_dir, "data\\datasets\\15min\\")
COINBASE_15MIN_PATH = os.path.join(current_dir, "data\\datasets\\15min\\coinbase\\coinbase_15min")
OKX_15MIN_PATH = os.path.join(current_dir, "data\\datasets\\15min\\okx\\okx_15min")

BACKTEST_DATA_PATH = os.path.join(current_dir, "data\\datasets\\15min\\scaled_backtest.csv")