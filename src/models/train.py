import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint

def _load_data(train_path: str,
               val_path: str,
               batch_size: int = 32) -> tuple[DataLoader, DataLoader]:
    '''
    Loads the training and validation data from the provided file paths into torch DataLoaders.
    
    Parameters:
        train_path (str): File path for the training data.
        val_path (str): File path for the validation data.
        batch_size (int): Batch size for the DataLoaders. (Default is 32.)
    
    Returns:
        tuple[DataLoader, DataLoader]: Tuple of training and validation DataLoaders.
    '''
    
    # Load training data
    train_data = pd.read_csv(train_path)
    x_train = train_data.iloc[:, :-1].values  # Features
    y_train = train_data.iloc[:, -1].values   # Labels

    # Load validation data
    val_data = pd.read_csv(val_path)
    x_val = val_data.iloc[:, :-1].values  # Features
    y_val = val_data.iloc[:, -1].values   # Labels
    
    # Wrap data as tensors
    train_dataset = TensorDataset(torch.tensor(x_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    val_dataset = TensorDataset(torch.tensor(x_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))

    # Create DataLoaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return (train_dataloader, val_dataloader)

def train_model(model, train_path: str, val_path: str):
    '''
    Trains the provided model and saves the best model.
    '''

    checkpoint_callback = ModelCheckpoint(
        monitor='val_acc',  # Metric to monitor
        save_top_k=1,       # Save only the best model
        mode='max',         # Maximize the metric
        dirpath='src\\models\\checkpoints\\',  # Directory to save checkpoints
        filename='best_model_{epoch:02d}_{val_loss:.2f}'  # Model file name
    )
    
    torch.set_float32_matmul_precision('high') # Performance optimization for 3060 Ti
    train_dataloader, val_dataloader = _load_data(train_path=train_path, val_path=val_path)
    trainer = Trainer(max_epochs=20, callbacks=[checkpoint_callback])
    trainer.fit(model, train_dataloader, val_dataloader)
