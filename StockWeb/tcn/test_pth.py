import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler #数据处理
import matplotlib.pyplot as plt #绘图
from Config import config #从自定义的Config.py参数文件中插入
from DataSplit import split_data #从自定义的DataSplit.py文件中导入split_data函数
#加载参数
config = config()

# 1.加载时间序列数据
df = pd.read_csv(config.data_path, index_col=0)
df = df.fillna(0.0)
# df = pd.read_csv(config.data_path, index_col=[0, 1], usecols=lambda x: x not in [0, 1])
# 2.将数据进行标准化
scaler = MinMaxScaler()
scaler_model = MinMaxScaler()
data = scaler_model.fit_transform(np.array(df))
scaler.fit_transform(np.array(df['close']).reshape(-1, 1))

# 3.从自定义的函数中获取训练数据
x_train, y_train, x_test, y_test = split_data(data, config.timestep, config.input_size)

# 4.将数据转为tensor
x_test_tensor = torch.from_numpy(x_test).to(torch.float32)
y_test_tensor = torch.from_numpy(y_test).to(torch.float32)
x_test_tensor = x_test_tensor.transpose(1, 2)

# 5.载入模型和参数
from model_TCN import TCN
model = TCN(config.input_size, config.output_size, config.num_channels, config.kernel_size, config.dropout)

model.load_state_dict(torch.load(config.save_path)) # 导入网络的参数,config.save_path中是网络参数的保存路径
first = 0
plot_size = 200
# 6.使用训练好的模型进行预测和可视化
model = model.eval() #消除dropout层的影响
y_test_pred = model(x_test_tensor)
# print(y_test_pred)
plt.figure(figsize=(12, 8))
plt.plot(scaler.inverse_transform(y_test_pred.detach().numpy().reshape(-1, 1)[first: plot_size]), "b") #展示部分预测效果
plt.plot(scaler.inverse_transform(y_test_tensor.detach().numpy().reshape(-1, 1)[first: plot_size]), "r") #展示部分预测效果
# plt.legend()
plt.show()
plt.close()
