import pandas as pd

def backtest(model,
             historical_data: pd.DataFrame,
             forward_window: int = 2,
             backward_window: int = 5,
             initial_capital: float = 10000.0,
             stop_loss: float = 0.1,
             device: str = 'cpu') -> None:
    '''
    Performs a backtest using the provided ML model to evaluate simulated trading performance
    in long and short positions.
    
    Parameters:
        model: The classification model to predict whether to 'buy', 'sell', or 'hold'
        historical_data (pd.DataFrame): The fully processed and normalized historical data sorted in temporal order by timestamp to backtest upon (long or short periods).
        forward_window (int): The number of days to look ahead for predictions. (default = 2)
        backward_window (int): The number of days to look back for feature extraction. (default = 5)
        initial_capital (float): The starting capital for the trading simulation. (default = 10000.0)
        stop_loss (float): The percentage of the price to use as a stop loss threshold. (default = 0.1)
        device (str): The device to perform inference on. (default = 'cpu')

    Returns:
        None
    '''

    capital = initial_capital
    position = 0 # 0 means no position
    trades = list() # Tracks all trades made

    # Init model
    model = model.to(device)

    initial_position = (forward_window + backward_window)
    for offset in range(0, len(historical_data) - initial_position, forward_window):
        
        # Get forward positions of both windows
        backward_window_front_pos = offset + backward_window
        forward_window_front_pos = backward_window_front_pos + forward_window

        # Grab the current window data
        backward_data = historical_data.iloc[offset:backward_window_front_pos]
        forward_data = historical_data.iloc[backward_window_front_pos:forward_window_front_pos]

if __name__ == "__main__":
    test_data = {
        "time": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    }
    test_df = pd.DataFrame(test_data)
    backtest(1, test_df)