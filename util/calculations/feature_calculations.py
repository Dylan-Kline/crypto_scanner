import pandas as pd
import numpy as np

def calculate_open_close_pct_change(data, open_price_column, close_price_column) -> pd.Series:
    """
    This function calculates the percentage change between open and close prices.
    
    Parameters:
        data (pd.DataFrame): DataFrame containing open and close prices
        open_price_column (str): Name of the column with open prices
        close_price_column (str): Name of the column with close prices
    
    Returns:
        pd.Series: a new Series containing the 'open_close_pct_change' with percentage changes.
    """

    close = data[close_price_column]
    open = data[open_price_column]
    open_close_pct = (close - open) / open

    return open_close_pct

