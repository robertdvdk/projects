"""
Author: Robert van der Klis

What does this module do

Usage: python3 ...
"""


# Import statements
import torch
import torch.nn as nn
import torch.nn.functional as F

# Function definitions
class AlexNet(nn.Module):
    def __init__(self):
        #TODO response normalization
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 96, (11, 11), 4), nn.ReLU(),
            nn.Conv2d(96, 256, (5, 5), 1, padding='same'), nn.ReLU(), nn.MaxPool2d(3, 2),
            nn.Conv2d(256, 384, (3, 3), 1, padding='same'), nn.ReLU(), nn.MaxPool2d(3, 2),
            nn.Conv2d(384, 384, (3, 3), 1, padding='same'), nn.ReLU(),
            nn.Conv2d(384, 256, (3, 3), 1, padding='same'), nn.ReLU(), nn.MaxPool2d(3, 2),
            nn.Flatten(),
            nn.Linear(4096, 4096), nn.ReLU(),
            nn.Linear(4096, 4096), nn.ReLU(),
            nn.Linear(4096, 1000)
        )

    def forward(self, x):
        return self.net(x)

def main():
    test = [[[[1 for i in range(224)] for i in range(224)] for i in range(3)] for i in range(32)]
    x = torch.Tensor(test)
    net = AlexNet()
    print(net.forward(x))


if __name__ == "__main__":
    main()
