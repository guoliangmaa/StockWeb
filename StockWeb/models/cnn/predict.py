import numpy as np
import torch
from pandas import DataFrame
from sklearn.preprocessing import MinMaxScaler

from StockWeb.utils.Config import config


def create_dataset(data, time_step):
    dataX, dataY = [], []
    for i in range(len(data) - time_step - 1):
        a = data[i:(i + time_step)]
        dataX.append(a)
        dataY.append(data[i + time_step])
    return np.array(dataX), np.array(dataY)


def cnn_predict(_config: config, origin_df: DataFrame):

    close_prices = origin_df['close'].values
    close_prices = close_prices.reshape(-1, 1)

    # 数据标准化
    scaler = MinMaxScaler(feature_range=(0, 1))
    close_prices = scaler.fit_transform(close_prices)

    time_step = _config.timestep  # 做了修改 原来直接设置为 10 现在用变量设置
    X, y = create_dataset(close_prices, time_step)
    X = torch.from_numpy(X).float().reshape(-1, 1, time_step)
    y = torch.from_numpy(y).float()

    X, y = create_dataset(close_prices, time_step)
    X = torch.from_numpy(X).float().reshape(-1, 1, time_step)
    y = torch.from_numpy(y).float()

    model = torch.load(_config.save_path).to(torch.device("cpu"))
    print(model)

    test_input = X[-1].reshape(1, 1, _config.timestep)
    predicted_price = model(test_input).detach().numpy().flatten()
    predicted_price = scaler.inverse_transform(predicted_price.reshape(-1, 1))

    return predicted_price
