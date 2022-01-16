import torch
import matplotlib.pyplot as plt
from data_reader import DataReader
from machine import Machine
import train
import test
import random

def gen_sal(data, y_true):
    criterion = torch.nn.MSELoss(reduction='mean')
    print("Generating saliency")
    model = torch.load("models/machine.h5")
    model.train()
    data = data.clone()
    data.requires_grad = True
    y_pred = model(data)
    loss = criterion(y_pred, y_true)
    print(f"Case Loss {loss.item():.6f}")
    loss.backward()
    x = torch.abs(data.grad)

    std = torch.std(x)
    mean = torch.mean(x)

    x = (x-mean)/std
    x = x - torch.min(x)
    x = x.squeeze().squeeze()
    plt.plot(x)
    plt.show()

def run_full_cycle():
    train.train()
    test.test()
    model = torch.load("models/machine.h5")
    model.eval()
    dr = DataReader()
    _, _, x_test, y_test = dr.get_data()
    x_test = x_test.reshape(x_test.shape[0], -1)
    y_test = y_test.reshape(-1, 1)

    index = random.randint(0, len(x_test) - 1)

    current_x = x_test[index].unsqueeze(dim=0)
    current_y = y_test[index].unsqueeze(dim=0)
    gen_sal(current_x, current_y)


if __name__ == "__main__":
    for i in range(5):
        run_full_cycle()