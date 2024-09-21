import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import nn
import lightning as L

class NN_algo_model(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.input = nn.Linear(128, 64)
        self.h1 = nn.Linear(64, 32)
        self.output = nn.Linear(32, 3)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)
        self.softmax = nn.Softmax(dim=1)
        self.criterion = nn.CrossEntropyLoss()
        
    def forward(self, x):
        x = self.leaky_relu(self.input(x))
        x = self.leaky_relu(self.h1(x))
        x = self.output(x)
        x = self.softmax(x)
        
        return x
    
class NN_Algo(L.LightningModule):
    '''
    This class builds off of the nn algo research paper.
    '''
    
    def __init__(self):
        super().__init__()
        self.model = NN_algo_model()
        
    def training_step(self, batch, batch_indx):
        x, y = batch
        logits = self.model(x)
        loss = self.model.criterion(logits, y)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_indx):
        x, y = batch
        logits = self.model(x)
        loss = self.model.criterion(logits, y)
        self.log('val_loss', loss)
        return loss
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=0.001)
        return optimizer
        
        