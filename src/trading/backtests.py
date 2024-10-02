import pandas as pd
import numpy as np

from ..models.predict import predict

def backtest(model,
             historical_data: pd.DataFrame,
             forward_window: int = 1,
             backward_window: int = 5,
             initial_capital: float = 10000.0,
             stop_loss: float = 0.1,
             device: str = 'cpu',
             exchange_fee: float = 0.001) -> None:
    '''
    Performs a backtest using the provided ML model to evaluate simulated trading performance
    in long and short positions.
    
    Parameters:
        model: The classification model to predict whether to 'buy', 'sell', or 'hold'
        historical_data (pd.DataFrame): The fully processed and normalized historical data sorted in temporal order by timestamp to backtest upon (long or short periods).
        forward_window (int): The number of days to look ahead for predictions. (default = 1)
        backward_window (int): The number of days to look back for feature extraction. (default = 5)
        initial_capital (float): The starting capital for the trading simulation. (default = 10000.0)
        stop_loss (float): The percentage of the price to use as a stop loss threshold. (default = 0.1)
        device (str): The device to perform inference on. (default = 'cpu')
        exchange_fee (float): The percentage fee applied on each trade. (default = 0.1%)

    Returns:
        None
    '''

    capital = initial_capital
    position = 0 # 0 means no position
    trades = list() # Tracks all trades made
    return_per_trade = list()
    open_price = 0.0
    buy_price = 0.0
    sell_price = 0.0

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

        # Grab price information
        close_price = forward_data['close'].values

        # Create feature vector
        input_vector = backward_data.values.flatten()

        # Predict next move using the model
        prediction = predict(model=model, data=input_vector)

        # Buy signal logic
        if prediction == 0 and position == 0:

            open_price = close_price
            buy_price = close_price * (1 + exchange_fee)
            position = capital / buy_price # Buy as much as possible
            print(f"Bought at {buy_price}, {position} units.")

        # Sell signal logic
        elif prediction == 2 and position > 0 and buy_price < (close_price * (1 - exchange_fee)):

            # Sell all assests
            sell_price = close_price * (1 - exchange_fee)
            capital = position * sell_price # Sell as much as possible

            # Calculate return of trade and profit
            returns = (sell_price - buy_price) / open_price
            return_per_trade.append(returns)
            profit = capital - initial_capital

            # Reset position
            position = 0

            print(f"Sold at price {sell_price}, Profit: {profit}, Return: {returns}, current_capital: {capital}")

        # Stop loss condition
        if position > 0 and close_price < (buy_price * (1 - stop_loss)):

            # Sell all assests
            sell_price = close_price * (1 - exchange_fee)
            capital = position * sell_price # Sell as much as possible

            # Calculate return of trade and profit
            returns = (sell_price - buy_price) / open_price
            return_per_trade.append(returns)
            profit = capital - initial_capital

            # Reset position
            position = 0

            print(f"Sold at stop-loss price {sell_price}, Profit: {profit}, Return: {returns}, current_capital: {capital}")

    # Handle the case where there is a position still open
    if position > 0:
        final_close_price = historical_data['close'].iloc[-1]
        sell_price = final_close_price * (1 - exchange_fee)

         # Calculate return of trade and profit
        returns = (sell_price - buy_price) / open_price
        return_per_trade.append(returns)
        profit = capital - initial_capital

        # Reset position
        position = 0

        print(f"Sold at price {sell_price}, Profit: {profit}, Return: {returns}, current_capital: {capital}")
    
    # Calculate total returns
    total_return = 1.0
    for r in return_per_trade:
        total_return *= (1 + r)

    # Calculate ROI
    roi = (capital - initial_capital) / initial_capital

    print(f"Final Capital: {capital}, Total Return: {total_return}, ROI: {roi}")
