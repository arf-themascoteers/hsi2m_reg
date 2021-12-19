import torch
import matplotlib.pyplot as plt
from data_reader import DataReader
from machine import Machine


def train():
    model = Machine()
    criterion = torch.nn.MSELoss(reduction='mean')
    optimiser = torch.optim.Adam(model.parameters(), lr=0.01)
    dr = DataReader()
    x_train, y_train, _, _ = dr.get_data()
    x_train = x_train.reshape(x_train.shape[0], -1)
    y_train = y_train.reshape(-1,1)
    for t in range(300):
        y_train_pred = model(x_train)
        loss = criterion(y_train_pred, y_train)
        #print("Epoch ", t, "MSE: ", loss.item())
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    print(f"Train Loss {loss.item():.2f}")
    torch.save(model, "models/machine2.h5")


def test():
    criterion = torch.nn.MSELoss(reduction='mean')
    model = torch.load("models/machine2.h5")

    dr = DataReader()
    _, _, x_test, y_test = dr.get_data()
    x_test = x_test.reshape(x_test.shape[0], -1)
    y_test = y_test.reshape(-1, 1)
    y_test_pred = model(x_test)
    loss = criterion(y_test_pred, y_test).item()
    print(f"Test Loss {loss:.2f}")
    plt.plot(y_test.squeeze().detach().numpy(), label = "Original")
    plt.plot(y_test_pred.squeeze().detach().numpy(), label = "Predicted")
    plt.legend()
    plt.show()


def gen_sal(data, y_true):
    print("Generating saliency")
    model = torch.load("models/machine2.h5")
    model.train()
    data = data.clone()
    data.requires_grad = True
    y_pred = model(data)
    loss = criterion(y_pred, y_true)
    loss.backward()
    x = torch.abs(data.grad)

    std = torch.std(x)
    mean = torch.mean(x)

    x = (x-mean)/std
    x = x - torch.min(x)
    x = x.squeeze().squeeze()
    plt.plot(x)
    plt.show()

if __name__ == "__main__":
    train()
    test()
    model = torch.load("models/machine2.h5")
    criterion = torch.nn.MSELoss(reduction='mean')
    model.eval()
    dr = DataReader()
    _, _, x_test, y_test = dr.get_data()
    x_test = x_test.reshape(x_test.shape[0], -1)
    y_test = y_test.reshape(-1, 1)
    items = 0

    for i in range(len(y_test)):
        current_x = x_test[i].unsqueeze(dim=0)
        current_y = y_test[i].unsqueeze(dim=0)
        y_test_pred = model(current_x)
        loss = criterion(y_test_pred, current_y).item()
        if loss < 0.1:
            gen_sal(current_x, current_y)
            break


