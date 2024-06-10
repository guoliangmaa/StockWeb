import numpy as np
import torch
from pandas import DataFrame
from sklearn.preprocessing import MinMaxScaler
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

from .lstm_model import LSTMModel
from StockWeb.utils.Config import config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_high = LSTMModel(1, 50, 2, 1)
model_low = LSTMModel(1, 50, 2, 1)

scalar_high = MinMaxScaler(feature_range=(0, 1))
scalar_low = MinMaxScaler(feature_range=(0, 1))

loss_high = nn.MSELoss()
loss_low = nn.MSELoss()

optimizer_high = torch.optim.Adam(model_high.parameters(), lr=0.001)
optimizer_low = torch.optim.Adam(model_low.parameters(), lr=0.001)


def create_dataset(data, time_step):
    dataX, dataY = [], []
    for i in range(len(data) - time_step - 1):
        a = data[i:(i + time_step)]
        dataX.append(a)
        dataY.append(data[i + time_step])
    return np.array(dataX), np.array(dataY)


def lstm_train_using_high_and_low(_config: config, df: DataFrame):
    # 设置随机种子
    torch.manual_seed(0)
    np.random.seed(0)

    # 把模型放到GPU上
    model_high.to(device)
    model_low.to(device)

    # 使用最高价和最低价的平均值
    high_price = df['high'].values.reshape(-1, 1)
    low_price = df['low'].values.reshape(-1, 1)

    # 数据标准化
    high_price = scalar_high.fit_transform(high_price)
    low_price = scalar_low.fit_transform(low_price)

    list_prices = [high_price, low_price]
    data_loader = []

    time_step = _config.timestep  # 做了修改 原来直接设置为 10 现在用变量设置

    for lst in list_prices:
        X, y = create_dataset(lst, time_step)
        X = torch.from_numpy(X).float().reshape(-1, time_step, 1)
        y = torch.from_numpy(y).float().reshape(-1, 1)
        # 创建数据加载器
        batch_size = 128
        dataset = TensorDataset(X, y)
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        data_loader.append(train_loader)

    # 训练模型
    num_epochs = _config.epochs  # 用变量替代

    for epoch in range(num_epochs):
        model_high.train()
        for batch_x, batch_y in data_loader[0]:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer_high.zero_grad()
            output = model_high(batch_x)
            loss = loss_high(output, batch_y)
            loss.backward()
            optimizer_high.step()
        if epoch % 10 == 0:
            print(f'Epoch {epoch} High Price Loss: {loss.item()}')

        model_low.train()
        for batch_x, batch_y in data_loader[1]:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer_low.zero_grad()
            output = model_low(batch_x)
            loss = loss_low(output, batch_y)
            loss.backward()
            optimizer_low.step()
        if epoch % 10 == 0:
            print(f'Epoch {epoch} Low Price Loss: {loss.item()}')


def lstm_predict(_config: config, df: DataFrame):
    # 使用最高价和最低价的平均值
    high_price = df['high'].values.reshape(-1, 1)
    low_price = df['low'].values.reshape(-1, 1)

    # 数据标准化
    high_price = scalar_high.transform(high_price)
    low_price = scalar_low.transform(low_price)

    list_prices = [high_price, low_price]
    data_loader = []

    time_step = _config.timestep  # 做了修改 原来直接设置为 10 现在用变量设置
    for lst in list_prices:
        X, y = create_dataset(lst, time_step)
        X = torch.from_numpy(X).float().reshape(-1, time_step, 1)
        y = torch.from_numpy(y).float().reshape(-1, 1)
        data_loader.append(X)

    model_high.to(torch.device("cpu"))
    model_low.to(torch.device("cpu"))

    predict_high_input = data_loader[0][-1].reshape(1, time_step, 1)
    predict_low_input = data_loader[1][-1].reshape(1, time_step, 1)

    predicted_high_price = model_high(predict_high_input).detach().numpy().flatten()
    predicted_high_price = scalar_high.inverse_transform(predicted_high_price.reshape(-1, 1))

    predicted_low_price = model_low(predict_low_input).detach().numpy().flatten()
    predicted_low_price = scalar_low.inverse_transform(predicted_low_price.reshape(-1, 1))

    # 进行大小判断 因为预测的高点不一定比预测的低点 高
    if predicted_low_price.item() > predicted_high_price.item():
        predicted_high_price, predicted_low_price = predicted_low_price, predicted_high_price

    return predicted_high_price.item(), predicted_low_price.item()
