import os
from dotenv import load_dotenv

load_dotenv()

FEATURES_EXCLUDED = ['ema_1', 'ema_20', 'ema_50', 'ema_100', 'timestamp']

current_dir = os.getcwd()

COINBASE_API_KEY = os.getenv("COINBASE_API_KEY")
COINBASE_API_SECRET = os.getenv("COINBASE_API_SECRET")
COINBASE_15MIN_PATH = os.path.join(current_dir, "data\\datasets\\15min\\coinbase\\coinbase_15min")

OKX_15MIN_PATH = os.path.join(current_dir, "data\\datasets\\15min\\okx\\okx_15min")