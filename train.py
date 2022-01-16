import torch
import torch.nn as nn
from machine import Machine
import time
import numpy as np
from data_reader import DataReader

def train():

    model = Machine()
    criterion = torch.nn.MSELoss(reduction='mean')
    optimiser = torch.optim.Adam(model.parameters(), lr=0.01)
    dr = DataReader()
    x_train, y_train, _, _ = dr.get_data()
    y_train = y_train.reshape(-1,1)
    for t in range(500):
        y_train_pred = model(x_train)
        loss = criterion(y_train_pred, y_train)
        print("Epoch ", t, "MSE: ", loss.item())
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    torch.save(model, "models/machine.h5")


if __name__ == "__main__":
    train()
