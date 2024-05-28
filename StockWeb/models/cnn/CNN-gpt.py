import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import datetime

# 设置随机种子
torch.manual_seed(0)
np.random.seed(0)

# 检查GPU是否可用，并设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

# 下载股票数据
start_time = time.time()
stock_code = '600776.SS'
start_date = '2015-01-01'
end_date = datetime.datetime.now().strftime('%Y-%m-%d')
stock_data = yf.download(stock_code, start=start_date, end=end_date)
print(f'Data loading time: {time.time() - start_time:.4f} seconds')

# 使用收盘价
close_prices = stock_data['Close'].values
close_prices = close_prices.reshape(-1, 1)

# 数据标准化
start_time = time.time()
scaler = MinMaxScaler(feature_range=(0, 1))
close_prices = scaler.fit_transform(close_prices)
print(f'Data preprocessing time: {time.time() - start_time:.4f} seconds')


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

# 创建数据加载器
batch_size = 64
dataset = TensorDataset(X, y)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


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


model = CNNModel().to(device)
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 训练模型
num_epochs = 100
start_time = time.time()
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad()
        output = model(batch_x)
        loss = loss_function(output, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    avg_epoch_loss = epoch_loss / len(train_loader)
    if epoch % 10 == 0:
        print(f'Epoch {epoch} Loss: {avg_epoch_loss:.4f}')

print(f'Training time: {time.time() - start_time:.4f} seconds')

# 预测
model.eval()
start_time = time.time()
test_input = X[-1].reshape(1, 1, time_step).to(device)
predicted_price = model(test_input).detach().cpu().numpy().flatten()
predicted_price = scaler.inverse_transform(predicted_price.reshape(-1, 1))
print(f'Prediction time: {time.time() - start_time:.4f} seconds')

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
