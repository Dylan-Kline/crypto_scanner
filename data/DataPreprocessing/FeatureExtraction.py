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
    raw_data['bullish_engulfing'] = talib.CDLENGULFING(raw_data['open'], raw_data['high'], raw_data['low'], raw_data['close']) > 0
    raw_data['bearish_engulfing'] = talib.CDLENGULFING(raw_data['open'], raw_data['high'], raw_data['low'], raw_data['close']) < 0
    raw_data['bullish_harami'] = talib.CDLHARAMI(raw_data['open'], raw_data['high'], raw_data['low'], raw_data['close']) > 0
    raw_data['bearish_harami'] = talib.CDLHARAMI(raw_data['open'], raw_data['high'], raw_data['low'], raw_data['close']) < 0
    raw_data['three_inside_down'] = talib.CDL3INSIDE(raw_data['open'], raw_data['high'], raw_data['low'], raw_data['close']) < 0
    raw_data['three_inside_up'] = talib.CDL3INSIDE(raw_data['open'], raw_data['high'], raw_data['low'], raw_data['close']) > 0
    raw_data['black_marubozu'] = talib.CDLMARUBOZU(raw_data['open'], raw_data['high'], raw_data['low'], raw_data['close']) < 0
    raw_data['white_marubozu'] = talib.CDLMARUBOZU(raw_data['open'], raw_data['high'], raw_data['low'], raw_data['close']) > 0
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
    raw_data['ema_1'] = talib.EMA(raw_data['close'], timeperiod=2)
    raw_data['ema_20'] = talib.EMA(raw_data['close'], timeperiod=20)
    raw_data['ema_50'] = talib.EMA(raw_data['close'], timeperiod=50)
    raw_data['ema_100'] = talib.EMA(raw_data['close'], timeperiod=100)

    raw_data['ema_1_20_crossover'] = np.where(raw_data['ema_1'] > raw_data['ema_20'], 1.0, 0.0)
    raw_data['ema_20_50_crossover'] = np.where(raw_data['ema_20'] > raw_data['ema_50'], 1.0, 0.0)
    raw_data['ema_50_100_crossover'] = np.where(raw_data['ema_50'] > raw_data['ema_100'], 1.0, 0.0)
    raw_data['ema_1_50_crossover'] = np.where(raw_data['ema_1'] > raw_data['ema_50'], 1.0, 0.0)

    # Temporal Features
    raw_data['month'] = pd.to_datetime(raw_data['timestamp']).dt.month
    raw_data['day'] = pd.to_datetime(raw_data['timestamp']).dt.day
    raw_data['num_samples_in_day'] = raw_data.groupby(pd.to_datetime(raw_data['timestamp']).dt.date).cumcount() + 1

    return raw_data