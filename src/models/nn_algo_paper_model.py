import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import nn
import lightning as L

class NN_algo_model(nn.Module):
    
    def __init__(self, input_size, output_size):
        super().__init__()
        self.input = nn.Linear(input_size, 128)
        self.h1 = nn.Linear(128, 64)
        self.h2 = nn.Linear(64, 32)
        self.output = nn.Linear(32, output_size)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)
        self.criterion = nn.CrossEntropyLoss()
        
    def forward(self, x):
        x = self.leaky_relu(self.input(x))
        x = self.leaky_relu(self.h1(x))
        x = self.leaky_relu(self.h2(x))
        x = self.output(x)
        
        return x
    
class NN_Algo(L.LightningModule):
    '''
    This class builds off of the nn algo research paper.
    '''
    
    def __init__(self, input_size, output_size):
        super().__init__()
        self.model = NN_algo_model(input_size=input_size, output_size=output_size)
        self.validation_accs = list()
        self.validation_steps_counter = 0
        
    def forward(self, x):
        return self.model(x)
        
    def training_step(self, batch, batch_indx):
        x, y = batch
        logits = self.model(x)
        loss = self.model.criterion(logits, y)

        # Step interval for logging training accuracy to console
        steps_interval = 350
        if self.global_step % steps_interval == 0:
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            acc = (preds == y).float().mean()
            print(acc)
        
        # Metric logging
        self.log('train_loss', loss)

        return loss
    
    def validation_step(self, batch, batch_indx):
        x, y = batch
        logits = self.model(x)
        loss = self.model.criterion(logits, y)

        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        acc = (preds == y).float().mean()
        self.validation_accs.append(acc)

        steps_interval = 100
        if self.validation_steps_counter % steps_interval == 0:
            print(y)
            print(preds)

        self.validation_steps_counter += 1
        self.log('val_loss', loss)
        self.log('val_acc', acc)
        return loss
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=0.0001)
        return optimizer
    
    def on_validation_epoch_end(self):
        max_acc = max(self.validation_accs)
        self.validation_accs.clear()
        self.validation_steps_counter = 0
        print(max_acc)
        
        