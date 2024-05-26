import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import datetime

# 下载股票数据
stock_code = '600776.SS'
start_date = '2015-01-01'
end_date = datetime.datetime.now().strftime('%Y-%m-%d')
stock_data = yf.download(stock_code, start=start_date, end=end_date)

# 使用收盘价
close_prices = stock_data['Close'].values
close_prices = close_prices.reshape(-1, 1)

# 数据标准化
scaler = MinMaxScaler(feature_range=(0, 1))
close_prices = scaler.fit_transform(close_prices)

# 创建数据集
def create_dataset(data, time_step):
    dataX, dataY = [], []
    for i in range(len(data) - time_step - 1):
        a = data[i:(i + time_step)]
        dataX.append(a)
        dataY.append(data[i + time_step])
    return np.array(dataX), np.array(dataY)

time_step = 10
X, y = create_dataset(close_prices, time_step)
X = torch.from_numpy(X).float().reshape(-1, 1, time_step)
y = torch.from_numpy(y).float()

# 定义 CNN 模型
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(32 * time_step // 2, 1)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

model = CNNModel()
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 训练模型
num_epochs = 50
for epoch in range(num_epochs):
    for i in range(len(X)):
        output = model(X[i:i+1])
        loss = loss_function(output, y[i:i+1])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if epoch % 10 == 0:
        print(f'Epoch {epoch} Loss: {loss.item()}')

# 预测
test_input = X[-1].reshape(1, 1, time_step)
predicted_price = model(test_input).detach().numpy().flatten()
predicted_price = scaler.inverse_transform(predicted_price.reshape(-1, 1))

# 绘制结果
plt.figure(figsize=(12, 6))
actual_data_count = len(close_prices)
predicted_index = np.arange(actual_data_count, actual_data_count + len(predicted_price))
plt.plot(scaler.inverse_transform(close_prices), label='Actual Prices')
plt.plot(predicted_index, predicted_price, 'ro-', label='Predicted Price')
plt.title('Stock Price Prediction with CNN')
plt.xlabel('Days')
plt.ylabel('Price')
plt.legend()
plt.show()
