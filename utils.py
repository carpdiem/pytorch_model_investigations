import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
import datasets as ds
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader
import torch.profiler as tp
import random
import matplotlib.pyplot as plt
import csv

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

def setup_mnist_dataloaders(batch_size = 100, device = 'cpu'):
    mnist = setup_mnist_datasets()
    train_loader = DataLoader(mnist["train"], batch_size=batch_size, shuffle=True, pin_memory=(device != 'cpu'))
    valid_loader = DataLoader(mnist["valid"], batch_size=batch_size, shuffle=False, pin_memory=(device != 'cpu'))
    test_loader = DataLoader(mnist["test"], batch_size=batch_size, shuffle=False, pin_memory=(device != 'cpu'))
    return {'train': train_loader, 'valid': valid_loader, 'test': test_loader}


class SimpleNet(nn.Module):
    def __init__(self, activation = None):
        super().__init__()
        self.layer1 = nn.Linear(28*28, 10)
        self.activation = activation

    def forward(self, x):
        x = self.layer1(x)
        if self.activation:
            x = self.activation(x)
        return x

class SimpleNet2(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(28*28, 10)
        self.layer2 = nn.Linear(10, 10)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x

class SimpleNet3(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(28*28, 14*14)
        self.relu1 = nn.ReLU()
        self.layer2 = nn.Linear(14*14, 10)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu1(x)
        x = self.layer2(x)
        return x

def training_loop(train_dl, validation_dl, model, loss_fn, optimizer, 
                  device='cpu',
                  lr = 1e-3,
                  verbose=False,
                  single_batch=False,
                  really_verbose=False):
    size = len(train_dl.dataset)
    optimizer = optimizer(model.parameters(), lr=lr)
    model.train()
    
    def train_batch(d):
        X = d['image'].view(-1, 28*28)
        y = d['label']
        X, y = X.to(device), y.to(device)

        # breakpoint()
        optimizer.zero_grad()
        pred = model(X)
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()
        
        return loss

    if single_batch:
        d = next(iter(train_dl))
        loss = train_batch(d)
        return loss
    else:
        for batch, d in enumerate(train_dl):
            loss = train_batch(d)

            if batch % 100 == 0:
                print(f"{batch} batches done")

            if verbose:
                if batch % 100 == 0:
                    if really_verbose:
                        for name, param in model.named_parameters():
                            if param.requires_grad:
                                grad_mean = torch.mean(param.grad)
                                grad_var = torch.var(param.grad)
                                grad_min = torch.min(param.grad)
                                grad_max = torch.max(param.grad)
                                print(f"param: {name} mean: {grad_mean} var: {grad_var} min: {grad_min} max: {grad_max}")
                    loss, current = loss.item(), (batch + 1) * len(X)
                    print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn, device='cpu', verbose=False):
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
    test_loss /= size
    correct /= size
    if verbose:
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return {'loss': test_loss, 'accuracy': correct}

def train_and_test(train_dl, validation_dl, test_dl, model,
                   loss_fn=nn.CrossEntropyLoss(),
                   optimizer=torch.optim.SGD,
                   lr = 1e-3,
                   device='cpu',
                   epochs=5,
                   verbose=False):
    optimizer = optimizer(model.parameters(), lr=lr)
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        training_loop(train_dl, validation_dl, model, loss_fn, optimizer, device, verbose)
        print("Performance on validation set:")
        test(validation_dl, model, loss_fn, device)
    print("Done!")
    print("Performance on test set:")
    test(test_dl, model, loss_fn, device)

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def profile_training_batch(train_dl, validation_dl, model, 
                           loss_fn=nn.CrossEntropyLoss(), 
                           optimizer=torch.optim.SGD, 
                           lr = 1e-3,
                           device='cpu', 
                           verbose=False,
                           num_batches=1):
    with tp.profile(activities=[tp.ProfilerActivity.CPU], record_shapes=True, with_flops=True) as profile:
        with tp.record_function("training_loop"):
            for i in range(num_batches):
                training_loop(train_dl, validation_dl, model, loss_fn, optimizer,
                              device = device,
                              lr = lr,
                              verbose = verbose,
                              single_batch=True)
    
    return profile

def print_profile(profile, sort_by = 'cpu_time_total', num_rows = 200):
    print(profile.key_averages(group_by_input_shape=True).table(sort_by=sort_by, row_limit=num_rows))

def set_seed(seed, deterministic=False):
    torch.use_deterministic_algorithms(deterministic)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

def loss_vs_flops(model, batch_size = 100, 
                         epochs = 0,
                         loss_fn = nn.CrossEntropyLoss(),
                         optimizer = torch.optim.SGD,
                         lr = 1e-3,
                         device = 'cpu',
                         include_dls = False,
                         verbose = False,
                         seed = 1729,
                         deterministic = False):
    ## set random seed if not False
    if seed is not False:
        set_seed(seed, deterministic=deterministic)

    mnist_dls = setup_mnist_dataloaders(batch_size = batch_size, device=device)
    
    def reset_model_parameters(model):
        for layer in model.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
            elif isinstance(layer, nn.Module):
                reset_model_parameters(layer)
    
    reset_model_parameters(model)

    ## first profile a single batch to determine flops / batch
    profile = profile_training_batch(mnist_dls['train'], mnist_dls['valid'], model,
                                     loss_fn = loss_fn,
                                     optimizer = optimizer,
                                     device = device,
                                     verbose = verbose,
                                     num_batches = 1)
    flops_per_batch = 0
    for event in profile.events():
        if event.flops is not None:
            flops_per_batch += event.flops

    ## now generate a training record
    ### First, we need to reset the model parameters
    
    reset_model_parameters(model)

    ### Second, we do the training, and record the validation losses
    batches_per_epoch = len(mnist_dls['train'])
    flops_per_epoch = flops_per_batch * batches_per_epoch

    losses = []
    accuracies = []
    for epoch in range(epochs):
        print(f"Epoch: {epoch + 1}")
        training_loop(mnist_dls['train'], mnist_dls['valid'], model, loss_fn, optimizer, device=device)
        test_res = test(mnist_dls['valid'], model, loss_fn, device=device)
        losses += [test_res['loss']]
        accuracies += [test_res['accuracy']]
        print(f"Validation Loss: {losses[-1]}")

    print(f"\nFinal Test Loss: {test(mnist_dls['test'], model, loss_fn, device=device)}")

    flops = [flops_per_epoch * (epoch + 1) for epoch in range(epochs)]

    return {'flops': np.array(flops), 
            'losses': np.array(losses),
            'accuracies': np.array(accuracies),
            'model': model}

def write_results_to_csv(res, path):
    flops = res['flops']
    losses = res['losses']
    accuracies = res['accuracies']

    if len(flops) != len(losses) or len(flops) != len(accuracies):
        raise ValueError("Lengths of flops, losses, and accuracies must be the same")
    
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['FLOPs', 'Loss', 'Accuracy'])
        for i in range(len(flops)):
            writer.writerow([flops[i], losses[i], accuracies[i]])

    return path

def plot_results(results, title = None, show = False, path = None):
    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('FLOPs')
    ax1.set_ylabel('Loss', color=color)
    ax1.plot(results['flops'], results['losses'], color=color)
    ax1.set_ylim(bottom=0)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:blue'
    ax2.set_ylabel('Accuracy', color=color)  # we already handled the x-label with ax1
    ax2.plot(results['flops'], results['accuracies'], color=color)
    ax2.set_ylim(top=1)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    if title:
        plt.title(title)

    if show:
        plt.show()

    if path:
        plt.savefig(path)

def plot_multiple_accuracies(results, titles, show = False, path = None):
    fig, ax1 = plt.subplots()

    for i, res in enumerate(results):
        ax1.plot(res['flops'], res['accuracies'], label = titles[i])

    ax1.set_xlabel('FLOPs')
    ax1.set_ylabel('Accuracy')
    ax1.set_ylim([0, 1])
    ax1.legend()

    fig.tight_layout()  # otherwise the right y-label is slightly clipped

    if show:
        plt.show()

    if path:
        plt.savefig(path)

def try_me(epochs = 3, plot=True):
    res = {'SimpleNet': loss_vs_flops(SimpleNet(), epochs = epochs),
           'SimpleNet2': loss_vs_flops(SimpleNet2(), epochs = epochs),
           'SimpleNet3': loss_vs_flops(SimpleNet3(), epochs = epochs)}
    
    if plot:
        plot_multiple_accuracies([res['SimpleNet'], res['SimpleNet2'], res['SimpleNet3']],
                                 ['SimpleNet', 'SimpleNet2', 'SimpleNet3'], show = True)
        
    return res
