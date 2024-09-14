import os
from dotenv import load_dotenv

load_dotenv()

current_dir = os.getcwd()

COINBASE_API_KEY = os.getenv("COINBASE_API_KEY")
COINBASE_API_SECRET = os.getenv("COINBASE_API_SECRET")
COINBASE_15MIN_PATH = os.path.join(current_dir, "data\\datasets\\15min\\coinbase\\coinbase_15min")

OKX_15MIN_PATH = os.path.join(current_dir, "data\\datasets\\15min\\okx\\okx_15min")