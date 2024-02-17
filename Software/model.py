import torch
from torch import nn

# speed breaks:
# air brakes
# roll assist (automatiquement par la roue)
# ground spoiler (kill the lift on the wing)

from torch import nn


class CovNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.act1 = nn.Tanh()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.act2 = nn.Tanh()
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(32 * 32 * 32, 1)
        self.regularize = nn.Sigmoid()

    def forward(self, input):
        out = self.pool1(self.act1(self.conv1(input)))
        out = self.pool2(self.act2(self.conv2(out)))
        out = out.view(-1, 32 * 32 * 32)
        out = self.fc1(out)
        return self.regularize(out)
