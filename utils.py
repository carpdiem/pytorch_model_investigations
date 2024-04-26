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
        "valid": test_valid["train"],
        "test": test_valid["test"]
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

def training_loop(train_dl, validation_dl, model, loss_fn, optimizer, device='cpu', verbose=False):
    size = len(train_dl.dataset)
    model.train()
    for batch, d in enumerate(train_dl):
        X = d['image'].view(-1, 28*28)
        y = d['label']
        X, y = X.to(device), y.to(device)

        optimizer.zero_grad()
        pred = model(X)
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            if verbose:
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        grad_mean = torch.mean(param.grad)
                        grad_var = torch.var(param.grad)
                        grad_min = torch.min(param.grad)
                        grad_max = torch.max(param.grad)
                        print(f"param: {name} mean: {grad_mean} var: {grad_var} min: {grad_min} max: {grad_max}")
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn, device='cpu'):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for d in dataloader:
            X = d['image'].view(-1, 28*28)
            y = d['label']
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

def train_and_test(model, train_dl, valid_dl, test_dl, loss_fn=nn.CrossEntropyLoss(), optimizer=torch.optim.SGD, device='cpu', epochs=5, verbose=False):
    optimizer = optimizer(model.parameters(), lr=1e-3)
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        training_loop(train_dl, valid_dl, model, loss_fn, optimizer, device, verbose)
        print("Performance on validation set:")
        test(valid_dl, model, loss_fn, device)
    print("Done!")
    print("Performance on test set:")
    test(test_dl, model, loss_fn, device)

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)