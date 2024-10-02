import pandas as pd
import numpy as np
import pickle
import os

from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import RobustScaler
from sklearn.utils import resample

from .FeatureExtraction import extract_features_OHLCV, remove_columns_processed_data
from .Label import label_crypto_AB
from .config import FEATURES_EXCLUDED, FEATURE_SCALER_PATH, CRYPTO_15MIN_PATH

def _balance_class_labels_undersample(labeled_data: pd.DataFrame) -> pd.DataFrame:
    '''
    Performs undersampling to balance the class labels of the provided data using imbalanced-learn library.

    Parameters:
        labeled_data (pd.DataFrame): The labeled dataset to balance.

    Returns:
        pd.DataFrame: A class balanced dataset.
    '''

    # Separate features and labels
    X = labeled_data.drop(columns=['label'])
    y = labeled_data['label']

    # Create RandomUnderSampler instance
    rus = RandomUnderSampler(random_state=42)

    # Apply undersampling
    X_resampled, y_resampled = rus.fit_resample(X, y)

    # Combine the resampled features and labels into a DataFrame
    balanced_data = pd.concat([pd.DataFrame(X_resampled, columns=X.columns), pd.DataFrame(y_resampled, columns=['label'])], axis=1)

    return balanced_data

def _extract_feature_names(df, label_col='label'):
    '''
    Extracts original feature names from the DataFrame, excluding the label column.
    
    Parameters:
        df (pd.DataFrame): The input DataFrame containing features and labels.
        label_col (str): The name of the label column to be excluded from features.
    
    Returns:
        list: A list of feature column names.
    '''
    feature_columns = df.columns[df.columns != label_col].tolist()
    return feature_columns

def _create_training_feature_vectors(data: pd.DataFrame,
                           backward_window: int = 5,
                           forward_window: int = 1) -> np.ndarray:
    '''
    Creates the feature vectorized data based on the backward and forward windows.

    Parameters:
        data (pd.DataFrame): The processed data to turn into feature vectors.
        backward_window (int): The number of days to look back and include as features. (default = 5)
        forward_window (int): The number of days to look ahead and include as targets. (default = 1)

    Returns:
        pd.DataFrame: A new dataframe containing the flattened feature vectors containing backward_window examples,
        and forward_window labels.
    '''
    
    # Create new feature names for all days worth of data up to backward_window size
    feature_columns = _extract_feature_names(data)  # Get all columns except 'label'
    new_feature_names = [f"{orig_name}_day_{i+1}" for i in range(backward_window) for orig_name in feature_columns]
    new_feature_names.append('label')

    examples = list()
    initial_position = backward_window + forward_window
    for offset in range(0, len(data) - initial_position + 1, forward_window):
        
        # Get forward positions of both windows
        backward_window_front_pos = offset + backward_window
        forward_window_front_pos = backward_window_front_pos + forward_window

        # Grab the current window data
        backward_data = data.iloc[offset:backward_window_front_pos, :-forward_window]
        forward_data = data.iloc[backward_window_front_pos:forward_window_front_pos, -forward_window:]

        # Create training example
        features = backward_data.values.flatten()
        labels = forward_data.values.flatten()
        example = np.concatenate([features, labels], axis=0)
        examples.append(example)

    # Create new training vector dataframe
    examples = pd.DataFrame(examples, columns=new_feature_names)

    return examples

def _split_data(raw_data: pd.DataFrame, 
                split_ratio: float = 0.7,
                backtest_start: str = '2022-01-01',
                backtest_end: str = '2022-04-01') -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits data into training and validation sets based on the specified ratio, after
    seperating the backtest data within the specified dates.
    
    Parameters:
        raw_data (pd.DataFrame): The input data to be split, must include 'timestamp' in yyyy-mm-dd format.
        split_ratio (float): The ratio for training data (default = 0.7).
        backtest_start (str): Start date of backtesting data (default = '2022-01-01')
        backtest_end (str): End date of backtesting data (default = '2022-04-01')

    Returns:
        pd.DataFrame, pd.DataFrame, pd.DataFrame: Training, validation, backtest.
    """
    backtest_data = raw_data[(raw_data['timestamp'] >= backtest_start) & (raw_data['timestamp'] <= backtest_end)]
    split_index = int(len(raw_data) * split_ratio)
    train_data = raw_data.iloc[:split_index]
    val_data = raw_data.iloc[split_index:]
    return train_data, val_data, backtest_data

# def create_test_dataset(raw_data: pd.DataFrame,
#                             candle_interval: str,
#                             backward_window: int = 5,
#                             forward_window: int = 1,
#                             scaler_path: str = os.path.join(FEATURE_SCALER_PATH, f"MLP_scaler.pkl")) -> pd.DataFrame:
#     '''
#     Converts the provided raw data into a test dataset that can be used for final model evaluation.
    
#     Parameters:
#         raw_data (pd.DataFrame): The raw data to be converted into a test dataset.
#         candle_interval (str): The candle interval the dataset contains.
#         backward_window (int): The number of days to look back in the data for the input vector. (default = 5)
#         forward_window (int): The number of days to predict the label for. (default = 1)
#         scaler_path (str): The path to the pre-fitted scaler to be used for scaling the test data.

#     Returns:
#         pd.DataFrame: A ready-made test dataset.
#     '''
#     interval_path = {
#         '15min':CRYPTO_15MIN_PATH
#     }
    
#     # Load pre-fitted scaler
#     with open(scaler_path, 'rb') as file:
#         scaler = pickle.load(file)
        
#     # Extract features
#     raw_data = extract_features_OHLCV(raw_data=raw_data)
#     raw_data = remove_columns_processed_data(raw_data, remove_columns=FEATURES_EXCLUDED)

#     # Scale the test data
#     raw_data_scaled = scaler.transform(raw_data)
#     raw_data_scaled = pd.DataFrame(raw_data_scaled, columns=raw_data.columns, index=raw_data.index)

#     # Label
#     labeled_data = label_crypto_AB(data=raw_data_scaled)

def training_pipeline_OHLCV(raw_data: pd.DataFrame,
                            candle_interval: str,
                            backward_window: int = 5,
                            forward_window: int = 1,
                            fit_scaler: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    '''
    Converts the provided raw OHLCV cryptocurrency data into a useable format for training.
    
    Parameters:
        raw_data (pd.DataFrame): The raw data to be converted into training data.
        candle_interval (str): The timespan between data points (e.g. '15min', '1h', etc.)
        backward_window (int): The number of days to look back in the data for the input vector. (default = 5)
        forward_window (int): The number of days to predict the label for. (default = 1)
        fit_scaler (bool): Whether to fit the scaler to the training data or not. (default = False)
        
    Returns:
        pd.DataFrame: Ready made training and validation datasets
    '''

    # Extract the features
    raw_data = extract_features_OHLCV(raw_data=raw_data)
    
    # Split data into training and validation sets
    train_data, val_data, backtest_data = _split_data(raw_data=raw_data)
    
    # Remove unneeded columns
    train_data = remove_columns_processed_data(train_data, remove_columns=FEATURES_EXCLUDED)
    val_data = remove_columns_processed_data(val_data, remove_columns=FEATURES_EXCLUDED)
    backtest_data = remove_columns_processed_data(backtest_data, remove_columns=FEATURES_EXCLUDED)

    # Grab/Create feature scaler
    scaler_path = os.path.join(FEATURE_SCALER_PATH, f"MLP_scaler.pkl")
    if os.path.exists(scaler_path) and not fit_scaler:

        with open(scaler_path, 'rb') as file:
            scaler = pickle.load(file)
            train_data_scaled = scaler.transform(train_data)
            val_data_scaled = scaler.transform(val_data)
            backtest_data_scaled = scaler.transform(backtest_data)

    else:
        scaler = RobustScaler()
        train_data_scaled = scaler.fit_transform(train_data)
        val_data_scaled = scaler.transform(val_data)
        backtest_data_scaled = scaler.transform(backtest_data)
        
        # Save scaler for later use
        with open(scaler_path, 'wb') as file:
            pickle.dump(scaler, file)

    # Label the datasets
    train_data_scaled = pd.DataFrame(train_data_scaled, columns=train_data.columns, index=train_data.index)
    val_data_scaled = pd.DataFrame(val_data_scaled, columns=val_data.columns, index=val_data.index)
    backtest_data_scaled = pd.DataFrame(backtest_data_scaled, columns=backtest_data.columns, index=backtest_data.index)
    
    train_labeled_data = label_crypto_AB(data=train_data_scaled)
    val_labeled_data = label_crypto_AB(data=val_data_scaled)

    # Create input feature vectors based on backward and forward windows
    train_vectored_data = _create_training_feature_vectors(data=train_labeled_data,
                                                           backward_window=backward_window,
                                                           forward_window=forward_window)
    
    val_vectored_data = _create_training_feature_vectors(data=val_labeled_data,
                                                         backward_window=backward_window,
                                                         forward_window=forward_window)

    # Balance the classes
    balanced_train_data = _balance_class_labels_undersample(labeled_data=train_vectored_data)

    # Save training data
    interval_path = {
        '15min':CRYPTO_15MIN_PATH
    }
    save_path_train = interval_path[candle_interval] + f"scaled_labeled_bWin_{backward_window}_fWin_{forward_window}_train.csv"
    save_path_val = interval_path[candle_interval] + f"scaled_labeled_bWin_{backward_window}_fWin_{forward_window}_val.csv"
    save_path_backtest = interval_path[candle_interval] + f"scaled_backtest.csv"
    
    balanced_train_data.to_csv(save_path_train, index=False)
    val_vectored_data.to_csv(save_path_val, index=False)
    backtest_data_scaled.to_csv(save_path_backtest, index=False)

    return balanced_train_data, val_vectored_data

def prediction_pipeline_OHLCV(raw_data: pd.DataFrame) -> pd.DataFrame:
    '''
    Converts the provided raw OHLCV cryptocurrency data into a usable format for the
    model to predict with, only for the most recent timestamp.
    
    Parameters:
        raw_data (pd.DataFrame): The raw data to be converted, which should contain at least 200 candles.
        
    Returns:
        pd.DataFrame: A fully processed dataset for prediction.
    '''
    
    # Extract the features and remove unnecessary columns
    raw_data = extract_features_OHLCV(raw_data=raw_data)
    raw_data = remove_columns_processed_data(raw_data, remove_columns=FEATURES_EXCLUDED)

    # Grab/Create feature scaler
    scaler_path = os.path.join(FEATURE_SCALER_PATH, f"MLP_scaler.pkl")
    if os.path.exists(scaler_path):

        with open(scaler_path, 'rb') as file:
            scaler = pickle.load(file)
            
    else:
        raise FileNotFoundError("There is no scaler used to train the model available.")
    
    # Scale all data
    raw_data_scaled = scaler.transform(raw_data)
    raw_data_scaled = pd.DataFrame(data=raw_data_scaled, columns=raw_data.columns)

    return raw_data_scaled
