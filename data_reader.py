import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime



class DataReader:
    def __init__(self):
        self.NUM_BANDS = 125
        self.raw_data = pd.read_csv("data/moisture.csv")
        self.moisture = torch.zeros(len(self.raw_data), dtype=torch.float32)
        self.bands = torch.zeros(len(self.raw_data), self.NUM_BANDS)
        for index, row in self.raw_data.iterrows():
            self.moisture[index] = row[2]
            self.bands[index] = torch.tensor(row[4:])

        self.moisture_scaler = MinMaxScaler(feature_range=(-1, 1))
        self.temperature_scaler = MinMaxScaler(feature_range=(-1, 1))

        temporary_moisture = self.moisture.reshape(-1,1)
        temporary_moisture = self.moisture_scaler.fit_transform(temporary_moisture)
        self.moisture = torch.tensor(temporary_moisture[:,0], dtype=torch.float32)

        self.data_size = len(self.moisture)

        test_set_size = int(0.3 * self.data_size)
        train_set_size = self.data_size - test_set_size

        self.x_train = self.bands[0:train_set_size]
        self.y_train = self.moisture[0:train_set_size]

        self.x_test = self.bands[train_set_size:]
        self.y_test = self.moisture[train_set_size:]

    def get_data(self):
        return self.x_train, self.y_train, self.x_test, self.y_test


if __name__ == "__main__":
    dr = DataReader()
    x_train, y_train, x_test, y_test = dr.get_data()
    print(len(x_train))
    print(len(y_train))
    print(len(x_test))
    print(len(y_test))
    print(len(dr.moisture))
    print(len(dr.bands))
    print(x_train[0])
    print(y_train[0])

