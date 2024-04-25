import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
import datasets as ds
import torchvision.transforms.functional as TF
import minai as mi
from torch.utils.data import DataLoader

def setup_mnist_datasets():
    mnist = ds.load_dataset("mnist")
    test_valid = mnist["test"].train_test_split(test_size=0.2)
    mnist = ds.DatasetDict({
        "train": mnist["train"],
        "test": test_valid["train"],
        "valid": test_valid["test"]
    })
    
    def tf(b):
        b['image'] = [TF.to_tensor(o) for o in b['image']]
        return b
    
    mnist = mnist.with_transform(tf)

    return mnist

def setup_mnist_dataloaders(batch_size = 64):
    mnist = setup_mnist_datasets()
    train_loader = DataLoader(mnist["train"], batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(mnist["valid"], batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(mnist["test"], batch_size=batch_size, shuffle=False)
    return {'train': train_loader, 'valid': valid_loader, 'test': test_loader}


class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(28*28, 10)

    def forward(self, x):
        x = self.layer1(x)
        return x