import torch
import torch.nn as nn

class Machine(nn.Module):
    def __init__(self):
        super(Machine, self).__init__()
        self.fc1 = nn.Linear(125,64)
        self.relu1 = nn.LeakyReLU()
        self.fc2 = nn.Linear(64, 16)
        self.relu2 = nn.LeakyReLU()
        self.fc3= nn.Linear(16, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x