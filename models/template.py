import torch
import torch.nn as nn

# standard params
#   'random_seed'
#   'learning_rate'
#   'batch size' 
#   'loss function'
#   'normalize yes/no'
#   'regularize'
#   'mixed precision yes/no'
#   'momentum'
#   'weight decay'
#   'dampening'
#   'beta'
#   'epsilon'
#   'prebuilt optimizer'
#   'learning rate scheduler function' https://docs.fast.ai/callback.schedule.html
 
params = {}

class DeepNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(28*28, 10)

    def forward(self, x):
        x = self.layer1(x)
        return x
