import torch
import torch.nn as nn

# standard params
#   'random_seed'
#   'learning_rate'
#   'optimizer'
#   'loss function'
#   something about... regularization?
params = {}

class DeepNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(28*28, 10)

    def forward(self, x):
        x = self.layer1(x)
        return x
