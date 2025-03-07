import pandas as pd
import numpy as np
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
    raw_data['three_white_soldiers'] = talib.CDL3WHITESOLDIERS(raw_data['open'], raw_data['high'], raw_data['low'], raw_data['close'])
    raw_data['doji'] = talib.CDLDOJI(raw_data['open'], raw_data['high'], raw_data['low'], raw_data['close'])
    raw_data['bullish_engulfing'] = np.where(talib.CDLENGULFING(raw_data['open'], raw_data['high'], raw_data['low'], raw_data['close']) > 0, 1.0, 0.0)
    raw_data['bearish_engulfing'] = np.where(talib.CDLENGULFING(raw_data['open'], raw_data['high'], raw_data['low'], raw_data['close']) < 0, 1.0, 0.0)
    raw_data['bullish_harami'] = np.where(talib.CDLHARAMI(raw_data['open'], raw_data['high'], raw_data['low'], raw_data['close']) > 0, 1.0, 0.0)
    raw_data['bearish_harami'] = np.where(talib.CDLHARAMI(raw_data['open'], raw_data['high'], raw_data['low'], raw_data['close']) < 0, 1.0, 0.0)
    raw_data['three_inside_down'] = np.where(talib.CDL3INSIDE(raw_data['open'], raw_data['high'], raw_data['low'], raw_data['close']) < 0, 1.0, 0.0)
    raw_data['three_inside_up'] = np.where(talib.CDL3INSIDE(raw_data['open'], raw_data['high'], raw_data['low'], raw_data['close']) > 0, 1.0, 0.0)
    raw_data['black_marubozu'] = np.where(talib.CDLMARUBOZU(raw_data['open'], raw_data['high'], raw_data['low'], raw_data['close']) < 0, 1.0, 0.0)
    raw_data['white_marubozu'] = np.where(talib.CDLMARUBOZU(raw_data['open'], raw_data['high'], raw_data['low'], raw_data['close']) > 0, 1.0, 0.0)
    raw_data['gravestone_doji'] = talib.CDLGRAVESTONEDOJI(raw_data['open'], raw_data['high'], raw_data['low'], raw_data['close'])
    raw_data['on_neck_pattern'] = talib.CDLONNECK(raw_data['open'], raw_data['high'], raw_data['low'], raw_data['close'])
    raw_data['hammer'] = talib.CDLHAMMER(raw_data['open'], raw_data['high'], raw_data['low'], raw_data['close'])
    raw_data['inverted_hammer'] = talib.CDLINVERTEDHAMMER(raw_data['open'], raw_data['high'], raw_data['low'], raw_data['close'])
    raw_data['morning_star'] = talib.CDLMORNINGSTAR(raw_data['open'], raw_data['high'], raw_data['low'], raw_data['close'])
    raw_data['evening_star'] = talib.CDLEVENINGSTAR(raw_data['open'], raw_data['high'], raw_data['low'], raw_data['close'])
    raw_data['piercing_line'] = talib.CDLPIERCING(raw_data['open'], raw_data['high'], raw_data['low'], raw_data['close'])
    raw_data['hanging_man'] = talib.CDLHANGINGMAN(raw_data['open'], raw_data['high'], raw_data['low'], raw_data['close'])
    raw_data['shooting_star'] = talib.CDLSHOOTINGSTAR(raw_data['open'], raw_data['high'], raw_data['low'], raw_data['close'])
    raw_data['dark_cloud_cover'] = talib.CDLDARKCLOUDCOVER(raw_data['open'], raw_data['high'], raw_data['low'], raw_data['close'])

    # Technical Indicators
    raw_data['upper_band'], raw_data['middle_band'], raw_data['lower_band'] = talib.BBANDS(raw_data['close'])
    raw_data['ULTOSC'] = talib.ULTOSC(raw_data['high'], raw_data['low'], raw_data['close'], timeperiod1=7, timeperiod2=14, timeperiod3=28)
    raw_data['RSI'] = talib.RSI(raw_data['close'], timeperiod=14)

    # Percentage price variation
    raw_data['close_pct_change'] = raw_data['close'].pct_change() * 100

    # Z-score for close price
    raw_data['close_zscore'] = (raw_data['close'] - raw_data['close'].rolling(window=30).mean()) / raw_data['close'].rolling(window=30).std()

    # Exponential Moving Average Crossovers
    raw_data['ema_1'] = talib.EMA(raw_data['close'], timeperiod=2).interpolate()
    raw_data['ema_20'] = talib.EMA(raw_data['close'], timeperiod=20).interpolate()
    raw_data['ema_50'] = talib.EMA(raw_data['close'], timeperiod=50).interpolate()
    raw_data['ema_100'] = talib.EMA(raw_data['close'], timeperiod=100).interpolate()

    raw_data.fillna(0.0, inplace=True)

    raw_data['ema_1_20_crossover'] = np.where(raw_data['ema_1'] > raw_data['ema_20'], 1.0, 0.0)
    raw_data['ema_20_50_crossover'] = np.where(raw_data['ema_20'] > raw_data['ema_50'], 1.0, 0.0)
    raw_data['ema_50_100_crossover'] = np.where(raw_data['ema_50'] > raw_data['ema_100'], 1.0, 0.0)
    raw_data['ema_1_50_crossover'] = np.where(raw_data['ema_1'] > raw_data['ema_50'], 1.0, 0.0)

    raw_data['ema_1_20_crossover'] = raw_data['ema_1_20_crossover'].diff()
    raw_data['ema_20_50_crossover'] = raw_data['ema_20_50_crossover'].diff()
    raw_data['ema_50_100_crossover'] = raw_data['ema_50_100_crossover'].diff()
    raw_data['ema_1_50_crossover'] = raw_data['ema_1_50_crossover'].diff()

    raw_data.fillna(0.0, inplace=True)

    # Temporal Features
    raw_data['month'] = pd.to_datetime(raw_data['timestamp']).dt.month
    raw_data['day'] = pd.to_datetime(raw_data['timestamp']).dt.day
    raw_data['num_samples_in_day'] = raw_data.groupby(pd.to_datetime(raw_data['timestamp']).dt.date).cumcount() + 1

    return raw_data

def remove_columns_processed_data(processed_data: pd.DataFrame, remove_columns: list['str'] | str, save_path: str = None) -> pd.DataFrame:
    '''
    Takes the feature extracted data and removes all unnecessary columns to fully create 
    the processed data copy for model and analytics usage.

    Parameters:
        feature_extracted_data (pd.DataFrame): Feature extracted data from which to remove columns.
        remove_columns (list[str] or str): The columns to remove from the provided data.

    Returns:
        pd.DataFrame containing all necessary columns for modelling.
    '''

    data = processed_data.drop(columns=remove_columns)

    if save_path:
        data.to_csv(save_path, index=False)

    return data