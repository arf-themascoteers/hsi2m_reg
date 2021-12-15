import torch
import torch.nn as nn

class Machine(nn.Module):
    def __init__(self):
        super(Machine, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(125,16),
            nn.LeakyReLU(),
            nn.Linear(16,1)
        )

    def forward(self, x):
        return self.fc(x)