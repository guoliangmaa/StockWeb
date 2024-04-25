import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset

# from Config import config  # 从自定义的Config.py参数文件中插入
from .DataSplit import split_data  # 从DataSplit.py文件中导入split_data函数
from .train import fit  # 从自定义的train.py文件中插入训练模板fit函数


# 加载参数
# config = config()


def train_model(config):
    import pandas as pd
    # 1.加载时间序列数据
    df = pd.read_csv(config.data_path, index_col=0)
    print(df)
    df = df.fillna(0.0)  # 把数据中的NA空值制成0
    # df = pd.read_csv(config.data_path, index_col=[0, 1], usecols=lambda x: x not in [0, 1])
    # 2.将数据进行标准化
    scaler = MinMaxScaler()
    scaler_model = MinMaxScaler()
    data = scaler_model.fit_transform(np.array(df))
    scaler.fit_transform(np.array(df['close']).reshape(-1, 1))

    # 3.从自定义的函数中获取训练数据
    x_train, y_train, x_test, y_test = split_data(data, config.timestep, config.input_size)

    # 4.将数据转为tensor
    x_train_tensor = torch.from_numpy(x_train).to(torch.float32)
    y_train_tensor = torch.from_numpy(y_train).to(torch.float32)
    x_test_tensor = torch.from_numpy(x_test).to(torch.float32)
    y_test_tensor = torch.from_numpy(y_test).to(torch.float32)
    x_train_tensor = x_train_tensor.transpose(1, 2)
    x_test_tensor = x_test_tensor.transpose(1, 2)
    # 5.形成训练数据集
    train_data = TensorDataset(x_train_tensor, y_train_tensor)
    test_data = TensorDataset(x_test_tensor, y_test_tensor)

    # 6.将数据加载成迭代器
    train_loader = torch.utils.data.DataLoader(train_data, config.batch_size, False)
    test_loader = torch.utils.data.DataLoader(test_data, config.batch_size, False)

    # 7.载入模型、定义损失、定义优化器
    from .model_TCN import TCN
    model = TCN(config.input_size, config.output_size, config.num_channels, config.kernel_size, config.dropout)

    loss_function = nn.MSELoss()  # 定义损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)  # 定义优化器

    # 创建存放loss参数的文件
    import pandas as pd
    # 创建loss.csv，记录loss
    df1 = pd.DataFrame(columns=['train_loss', 'test_loss'])  # 列名
    df1.to_csv("./loss.csv", index=False)  # 路径可以根据需要更改

    # 8.开始训练
    train_loss = []
    test_loss = []
    bst_loss = np.inf
    for epoch in range(config.epochs):
        epoch_loss, epoch_test_loss = fit(epoch, model, loss_function, optimizer, train_loader, test_loader, bst_loss,
                                          config)
        # 将loss存进csv文件中
        list = [epoch_loss, epoch_test_loss]  # 创建存放loss的列表
        data = pd.DataFrame([list])
        data.to_csv("./loss.csv", mode='a', header=False, index=False)
        train_loss.append(epoch_loss)
        test_loss.append(epoch_test_loss)
    print('Finished Training')

    if config.show_figure:
        # 9.损失可视化
        plt.plot(range(1, config.epochs + 1), train_loss, label='train_loss')
        plt.plot(range(1, config.epochs + 1), test_loss, label='test_loss')
        plt.legend()
        plt.show()

        # 10.显示预测结果--train
        plot_size = 200  # 取前200个点进行观察
        plt.figure(figsize=(12, 8))
        model.eval()
        with torch.no_grad():
            plt.plot(scaler.inverse_transform((model(x_train_tensor).detach().numpy().reshape(-1, 1)[: plot_size])),
                     "b")
            plt.plot(scaler.inverse_transform(y_train_tensor.detach().numpy().reshape(-1, 1)[: plot_size]), "r")
            plt.legend()
            plt.show()

        # 10.显示预测结果--test
        model.eval()  # 消除dropout层的影响
        with torch.no_grad():
            y_test_pred = model(x_test_tensor)
            plt.figure(figsize=(12, 8))
            plt.plot(scaler.inverse_transform(y_test_pred.detach().numpy().reshape(-1, 1)[: plot_size]), "b")
            plt.plot(scaler.inverse_transform(y_test_tensor.detach().numpy().reshape(-1, 1)[: plot_size]), "r")
            plt.legend()
            plt.show()

# train_model()
