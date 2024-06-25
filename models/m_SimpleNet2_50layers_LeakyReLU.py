import torch
import torch.nn as nn

default_activation = lambda : nn.LeakyReLU(negative_slope=0.01)

class DeepNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.initial_activation = default_activation()
        self.num_middle_layers = 50
        self.layer1 = nn.Linear(28*28, 10*10)
        self.middle_layers = nn.ModuleList([nn.Linear(10*10, 10*10) for i in range(self.num_middle_layers)])
        self.activations = nn.ModuleList([default_activation() for i in range(self.num_middle_layers)])
        self.layerLast = nn.Linear(10*10, 10)
        

    def forward(self, x):
        x = self.layer1(x)
        x = self.initial_activation(x)
        for middle_layer, activation in zip(self.middle_layers, self.activations):
            x = middle_layer(x)
            x = activation(x)
        x = self.layerLast(x)
        return x