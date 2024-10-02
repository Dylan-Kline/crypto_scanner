import pandas as pd
import numpy as np
import talib

from src.util.calculations.feature_calculations import calculate_open_close_pct_change

def label_crypto_AB(data: pd.DataFrame, forW: int = 2, backW: int = 5, 
                    close_price_column: str = 'close',
                    open_price_column: str = 'open',
                    fee: float = 0.015,
                     save_path: str = None) -> pd.DataFrame:
    '''
    Labels the provided data with buy, hold, sell based on alpha/beta thresholds.

    Parameters:
        data (pd.DataFrame): The unlabeled data to be labeled.
        forW (int): Forward window size.
        backW (int): Backward window size.
        close_price_column (str): The column name which contains the close prices.
        open_price_column (str): The column name which contains the open prices.
        save_path (str): Optional path for saving labeled data.

    Returns:
        pd.DataFrame: Labeled data with a column 'labels' with values 'buy', 'hold', 'sell'
    '''

    # Target Classes for labels
    trade_options = {
        'buy':0,
        'hold':1,
        'sell':2
    }

    # Set thresholds
    open_close_pct = calculate_open_close_pct_change(data=data, open_price_column=open_price_column, close_price_column=close_price_column)
    
    alpha = np.percentile(open_close_pct, 85.0)
    beta = np.percentile(open_close_pct, 99.7)
    beta += (forW - 1) * (.1 * beta)

    # Compute the EMA of close prices using backW
    ema = talib.EMA(data[close_price_column], timeperiod=backW)
    updated_close_prices = pd.Series(data= ema, index=data[close_price_column].index, dtype="float64")
    #updated_close_prices.fillna(data[close_price_column], inplace=True)
    
    # Initialize labels
    labels = pd.Series(index=data[close_price_column].index, dtype="object")
    
    for t in range(len(data) - forW):

        open_T = updated_close_prices.iloc[t]
        close_TK = updated_close_prices.iloc[t + forW]

        # Compute the return of close prices over the next forW period
        #print(close_TK, open_T)
        future_return = ((1 - fee) * close_TK - (1 + fee) * open_T) / open_T
        #print(future_return)
        
        # Detemine trade option
        if alpha < abs(future_return) < beta:
            if future_return > 0:
                labels.iloc[t] = trade_options['buy']
            else:
                labels.iloc[t] = trade_options['sell']
        else:
            labels.iloc[t] = trade_options['hold']

    labels.fillna(1, inplace=True)
    data['label'] = labels
    
    return data







