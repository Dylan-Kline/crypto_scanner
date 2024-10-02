import pandas as pd
import numpy as np
import pickle
import os

from sklearn.preprocessing import RobustScaler
from sklearn.utils import resample

from .FeatureExtraction import extract_features_OHLCV, remove_columns_processed_data
from .Label import label_crypto_AB
from .config import FEATURES_EXCLUDED, FEATURE_SCALER_PATH, CRYPTO_15MIN_PATH

def _balance_class_labels_undersample(labeled_data: pd.DataFrame) -> pd.DataFrame:
    '''
    Performs undersampling to balance the class labels of the provided data.

    Parameters:
        labeled_data (pd.DataFrame): The labeled dataset to balance.

    Returns:
        pd.DataFrame: A class balanced dataset.
    '''

    # Balance the classes
    buy_data = labeled_data[labeled_data['label'] == 0]
    hold_data = labeled_data[labeled_data['label'] == 1]
    sell_data = labeled_data[labeled_data['label'] == 2]

    majority_undersampled = resample(hold_data, replace=False,
                                     n_samples=len(buy_data),
                                     random_state=42)
    balanced_data = pd.concat([buy_data, majority_undersampled, sell_data])

    return balanced_data

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

    examples = list()
    initial_position = backward_window + forward_window
    for offset in range(0, len(data) - backward_window, forward_window):
        
        # Get forward positions of both windows
        backward_window_front_pos = offset + backward_window
        forward_window_front_pos = backward_window_front_pos + forward_window

        # Grab the current window data
        backward_data = data.iloc[offset:backward_window_front_pos, :-forward_window]
        forward_data = data.iloc[backward_window_front_pos:forward_window_front_pos, -forward_window:]

        print(backward_data)
        print(forward_data)

        # Create training example
        features = backward_data.values.flatten()
        labels = forward_data.values.flatten()
        print(features)
        print(labels)

        example = np.concatenate([features, labels], axis=0)
        examples.append(example)

    print(pd.DataFrame(examples))

    return examples

def training_pipeline_OHLCV(raw_data: pd.DataFrame,
                            candle_interval: str,
                            backward_window: int = 5,
                            forward_window: int = 1) -> pd.DataFrame:
    '''
    Converts the provided raw OHLCV cryptocurrency data into a useable format for training.
    
    Parameters:
        raw_data (pd.DataFrame): The raw data to be converted into training data.
        candle_interval (str): The timespan between data points (e.g. '15min', '1h', etc.)
        backward_window (int): The number of days to look back in the data for the input vector. (default = 5)
        forward_window (int): The number of days to predict the label for. (default = 1)

    Returns:
        pd.DataFrame: A ready-made training dataset.
    '''

    # Extract the features
    raw_data = extract_features_OHLCV(raw_data=raw_data)
    raw_data = remove_columns_processed_data(raw_data, remove_columns=FEATURES_EXCLUDED)

    # Grab/Create feature scaler
    scaler_path = os.path.join(FEATURE_SCALER_PATH, f"MLP_scaler.pkl")
    if os.path.exists(scaler_path):

        with open(scaler_path, 'rb') as file:
            scaler = pickle.load(file)
            raw_data_scaled = scaler.transform(raw_data)

    else:
        scaler = RobustScaler()
        raw_data_scaled = scaler.fit_transform(raw_data)
        
        # Save scaler for later use
        with open(scaler_path, 'wb') as file:
            pickle.dump(scaler, file)

    # Label the data
    raw_data_scaled = pd.DataFrame(raw_data_scaled, columns=raw_data.columns)
    labeled_data = label_crypto_AB(data=raw_data_scaled)

    # Create input feature vectors based on backward window and labels from forward window
    vectored_data = _create_training_feature_vectors(data=labeled_data,
                                                    backward_window=backward_window,
                                                    forward_window=forward_window)

    # Balance the classes
    balanced_data = _balance_class_labels_undersample(labeled_data=vectored_data)

    # Save training data
    interval_path = {
        '15min':CRYPTO_15MIN_PATH
    }
    save_path = interval_path[candle_interval] + "scaled_labeled.csv"
    balanced_data.to_csv(save_path, index=False)

    return balanced_data

def prediction_pipeline_OHLCV(raw_data: pd.DataFrame) -> pd.DataFrame:
    '''
    Converts the provided raw OHLCV cryptocurrency data into a usable format for the
    model to predict with, only for the most recent timestamp.
    
    Parameters:
        raw_data (pd.DataFrame): The raw data to be converted, which should contain at least 200 candles.
        
    Returns:
        pd.DataFrame: A fully processed dataset for prediction, containing only the most recent timestamp.
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
    
    # Select the most recent candle to use for prediction
    latest_data = raw_data_scaled[:-1].copy()
    
    return latest_data
