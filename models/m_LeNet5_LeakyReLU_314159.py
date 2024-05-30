import torch
import torch.nn as nn

class DeepNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2)
        self.act1 = nn.LeakyReLU(negative_slope=0.01)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0)
        self.act2 = nn.LeakyReLU(negative_slope=0.01)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(16, 120, kernel_size=5, stride=1, padding=0)
        self.act3 = nn.LeakyReLU(negative_slope=0.01)

        self.flat = nn.Flatten()
        self.fc1 = nn.Linear(120, 84)
        self.act4 = nn.LeakyReLU(negative_slope=0.01)
        self.fc2 = nn.Linear(84, 10)

    def forward(self, x):
        x = torch.reshape(x, (-1, 1, 28, 28))
        x = self.act1(self.conv1(x))
        x = self.pool1(x)
        x = self.act2(self.conv2(x))
        x = self.pool2(x)
        x = self.act3(self.conv3(x))
        x = self.act4(self.fc1(self.flat(x)))
        x = self.fc2(x)
        return x
