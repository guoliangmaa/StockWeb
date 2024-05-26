import numpy as np
import torch
from pandas import DataFrame
from sklearn.preprocessing import MinMaxScaler
from torch import nn

from .lstm_model import LSTMModel
from StockWeb.utils.Config import config

def create_dataset(data, time_step):
    dataX, dataY = [], []
    for i in range(len(data) - time_step - 1):
        a = data[i:(i + time_step)]
        dataX.append(a)
        dataY.append(data[i + time_step])
    return np.array(dataX), np.array(dataY)


def lstm_train(_config: config, origin_df: DataFrame):
    # 设置随机种子
    torch.manual_seed(0)
    np.random.seed(0)

    # 使用收盘价
    close_prices = origin_df['close'].values
    close_prices = close_prices.reshape(-1, 1)

    # 数据标准化
    scaler = MinMaxScaler(feature_range=(0, 1))
    close_prices = scaler.fit_transform(close_prices)

    time_step = _config.timestep  # 做了修改 原来直接设置为 10 现在用变量设置
    X, y = create_dataset(close_prices, time_step)
    X = torch.from_numpy(X).float().reshape(-1, time_step, 1)
    y = torch.from_numpy(y).float().reshape(-1, 1)

    model = LSTMModel(1, 50, 2, 1)
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 训练模型
    num_epochs = _config.epochs  # 用变量替代
    for epoch in range(num_epochs):
        for i in range(len(X)):
            optimizer.zero_grad()
            output = model(X[i:i + 1])
            loss = loss_function(output, y[i])
            loss.backward()
            optimizer.step()
        if epoch % 10 == 0:
            print(f'Epoch {epoch} Loss: {loss.item()}')

    torch.save(model, _config.save_path)