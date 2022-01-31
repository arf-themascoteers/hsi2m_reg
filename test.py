import torch
from data_reader import DataReader

def test():
    model = torch.load("models/machine.h5")
    criterion = torch.nn.MSELoss(reduction='mean')
    dr = DataReader()
    _, _, x_test, y_test = dr.get_data()
    y_test = y_test.reshape(-1,1)
    y_test_pred = model(x_test)
    loss = criterion(y_test_pred, y_test)
    print("Test MSE: ", loss.item())

if __name__ == "__main__":
    test()
