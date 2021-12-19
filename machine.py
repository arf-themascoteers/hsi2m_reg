import torch
import torch.nn as nn

class Machine(nn.Module):
    def __init__(self):
        super(Machine, self).__init__()
        self.fc1 = nn.Linear(125,16)
        self.relu1 = nn.LeakyReLU()
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        return x