import pandas as pd
import talib

def extract_features_OHLCV(raw_data: pd.DataFrame) -> pd.DataFrame:
    '''
    Extract technical indicators and features from OHLCV data.
    
    Parameters:
        raw_data (pd.DataFrame): DataFrame containing raw OHLC data.
    
    Returns:
        pd.DataFrame: DataFrame with added technical indicators.
    '''

    # Candlestick patterns
    raw_data['three_black_crows'] = talib.CDL3BLACKCROWS(raw_data['open'], raw_data['high'], raw_data['low'], raw_data['close'])

    return raw_data