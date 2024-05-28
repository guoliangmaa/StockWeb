import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from pandas_datareader import data as pdr
import datetime
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

# 设置随机种子
torch.manual_seed(0)
np.random.seed(0)

# 检查GPU是否可用，并设置设备
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
print(f'Using device: {device}')

# 下载股票数据
stock_code = '600776.SS'
start_date = '2020-01-01'
end_date = datetime.datetime.now().strftime('%Y-%m-%d')
stock_data = yf.download(stock_code, start=start_date, end=end_date)

# 使用收盘价
close_prices = stock_data['Close'].values
close_prices = close_prices.reshape(-1, 1)

# 数据标准化
begin_time = time.time()
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
X = torch.from_numpy(X).float().reshape(-1, time_step, 1)
y = torch.from_numpy(y).float().reshape(-1, 1)

# 创建数据加载器
batch_size = 128
dataset = TensorDataset(X, y)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 定义 LSTM 模型
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_layer_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, input_seq):
        lstm_out, _ = self.lstm(input_seq)
        predictions = self.linear(lstm_out[:, -1])
        return predictions

model = LSTMModel(1, 50, 2, 1).to(device)

loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 训练模型
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad()
        output = model(batch_x)
        loss = loss_function(output, batch_y)
        loss.backward()
        optimizer.step()
    if epoch % 10 == 0:
        print(f'Epoch {epoch} Loss: {loss.item()}')

# 预测
model.eval()
test_input = X[-1].reshape(1, time_step, 1).to(device)
predicted_price = model(test_input).detach().cpu().numpy().flatten()
predicted_price = scaler.inverse_transform(predicted_price.reshape(-1, 1))

# 绘制结果
plt.figure(figsize=(12, 6))
actual_price_plot = scaler.inverse_transform(close_prices)
plt.plot(actual_price_plot, label='Actual Prices')

predicted_index = np.arange(len(close_prices), len(close_prices) + 1)
plt.plot(predicted_index, predicted_price, marker='o', label='Predicted Price', color='red')

plt.title('Stock Price Prediction of Dongfang Tong')
plt.xlabel('Days')
plt.ylabel('Price')
plt.legend()
plt.show()

# 记录结束时间
end_time = time.time()
elapsed_time = end_time - begin_time
print(f"Elapsed time: {elapsed_time:.4f} seconds")
