import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint

def _load_data(file_path: str) -> tuple[DataLoader, DataLoader]:
    '''
    Loads the data from the provided file path into a torch DataLoader.
    
    Parameters:
        file_path (str): File path for the data to be loaded.
    
    Returns:
        tuple[DataLoader, DataLoader]: Tuple of training and validation dataloaders.
    '''
    
    loaded_data = pd.read_csv(file_path)
    x = loaded_data.iloc[:, :-1].values
    y = loaded_data.iloc[:, -1].values
    
    # Grab examples and labels
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.3, random_state=42)
    
    # Wrap data as tensors
    train_dataset = TensorDataset(torch.tensor(x_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    val_dataset = TensorDataset(torch.tensor(x_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))
    
    # Create DataLoaders
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    return (train_dataloader, val_dataloader)

def train_model(model, data_path: str):
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
    train_dataloader, val_dataloader = _load_data(file_path=data_path)
    trainer = Trainer(max_epochs=20, callbacks=[checkpoint_callback])
    trainer.fit(model, train_dataloader, val_dataloader)
