import pandas as pd
import pickle
import os

from sklearn.preprocessing import RobustScaler
from sklearn.utils import resample

from .FeatureExtraction import extract_features_OHLCV, remove_columns_processed_data
from .Label import label_crypto_AB
from .config import FEATURES_EXCLUDED, FEATURE_SCALER_PATH, CRYPTO_15MIN_PATH

# TODO
# 1) create prediction pipeline

def training_pipeline_OHLCV(raw_data: pd.DataFrame,
                            candle_interval: str) -> pd.DataFrame:
    '''
    Converts the provided raw OHLCV cryptocurrency data into a useable format for training.
    
    Parameters:
        raw_data (pd.DataFrame): The raw data to be converted into training data.
        candle_interval (str): The timespan between data points (e.g. '15min', '1h', etc.)
        
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

    # Balanace the classes
    buy_data = labeled_data[labeled_data['label'] == 0]
    hold_data = labeled_data[labeled_data['label'] == 1]
    sell_data = labeled_data[labeled_data['label'] == 2]

    majority_undersampled = resample(hold_data, replace=False,
                                     n_samples=len(buy_data),
                                     random_state=42)
    balanced_data = pd.concat([buy_data, majority_undersampled, sell_data])

    # Save training data
    interval_path = {
        '15min':CRYPTO_15MIN_PATH
    }
    save_path = interval_path[candle_interval] + "scaled_labeled.csv"
    balanced_data.to_csv(save_path, index=False)

    return labeled_data

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