import torch
import torch.nn as nn

class DeepNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_middle_layers = 1
        self.layer1 = nn.Linear(28*28, 10*10)
        self.middle_layers = nn.ModuleList([nn.Linear(10*10, 10*10) for i in range(self.num_middle_layers)])
        self.layerLast = nn.Linear(10*10, 10)
        

    def forward(self, x):
        x = self.layer1(x)
        for middle_layer in self.middle_layers:
            x = middle_layer(x)
        x = self.layerLast(x)
        return x