import numpy as np

#划分数据集
def split_data(data, timestep, feature_size):
    dataX = []  # 保存X
    dataY = []  # 保存Y

    # 每个个滑动窗口的数据保存到X中，将窗口未来的一天保存到Y中
    for index in range(len(data) - timestep):
        dataX.append(data[index: index + timestep])#选取6列的特征数据
        dataY.append(data[index + timestep][1]) #标签，#选取收盘价列作为标签
        # print(len(dataX))
        # print(len(dataY))

    #数据格式转换
    dataX = np.array(dataX)
    dataY = np.array(dataY)

    # 获取训练集大小
    train_size = int(np.round(0.8 * dataX.shape[0]))

    # 划分训练集、测试集
    x_train = dataX[: train_size, :].reshape(-1, timestep, feature_size)
    y_train = dataY[: train_size].reshape(-1, 1)

    x_test = dataX[train_size:, :].reshape(-1, timestep, feature_size)
    y_test = dataY[train_size:].reshape(-1, 1)

    return [x_train, y_train, x_test, y_test]