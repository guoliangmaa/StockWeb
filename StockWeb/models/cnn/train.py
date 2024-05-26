import numpy as np
import torch
import torch.nn as nn
from pandas import DataFrame
from sklearn.preprocessing import MinMaxScaler

from .CNNModel import CNNModel
from StockWeb.utils.Config import config


# 创建数据集
def create_dataset(data, time_step):
    dataX, dataY = [], []
    for i in range(len(data) - time_step - 1):
        a = data[i:(i + time_step)]
        dataX.append(a)
        dataY.append(data[i + time_step])
    return np.array(dataX), np.array(dataY)


def cnn_train(_config: config, origin_df: DataFrame):
    # 使用收盘价
    close_prices = origin_df['close'].values
    close_prices = close_prices.reshape(-1, 1)

    # 数据标准化
    scaler = MinMaxScaler(feature_range=(0, 1))
    close_prices = scaler.fit_transform(close_prices)

    time_step = _config.timestep  # 做了修改 原来直接设置为 10 现在用变量设置
    X, y = create_dataset(close_prices, time_step)
    X = torch.from_numpy(X).float().reshape(-1, 1, time_step)
    y = torch.from_numpy(y).float()

    model = CNNModel()
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
    num_epochs = _config.epochs  # 与原本做了修改 用变量代替
    for epoch in range(num_epochs):
        for i in range(len(X)):
            output = model(X[i:i + 1])
            loss = loss_function(output, y[i:i + 1])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if epoch % 10 == 0:
            print(f'Epoch {epoch} Loss: {loss.item()}')

    torch.save(model, _config.save_path)
