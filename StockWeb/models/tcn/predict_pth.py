import torch
import numpy as np
from pandas import DataFrame
from sklearn.preprocessing import MinMaxScaler  # 数据处理
from StockWeb.utils.Config import config as Config  # 从自定义的Config.py参数文件中插入
from .model_TCN import TCN

# 加载参数

# config = config()
# 加载时间序列数据
# df = pd.read_csv(config.data_path, index_col=0)
# df = df.fillna(0.0)  # 把数据中的NA空值制成0


def predict(df: DataFrame, config: Config):
    # 将数据进行标准化
    scaler = MinMaxScaler()
    scaler_model = MinMaxScaler()
    data = scaler_model.fit_transform(np.array(df))
    scaler.fit_transform(np.array(df['close']).reshape(-1, 1))

    # 取数据中前50天的6个特征数据输入模型进行预测第51天的股票收盘价，可自行修改范围
    # 因为数据已经是从头开始的了 所以应该是后50个
    test_ = data[-config.timestep:]
    # reshape成适合模型输入的格式
    test = test_.reshape((1, config.timestep, config.input_size))
    # 其真实标签是50天后1天数据股票收盘价 小马注释
    # label = data[config.timestep][1]
    # 转化成torch_tensor以便于输入模型
    test_tensor = torch.from_numpy(test).to(torch.float32)
    test_tensor = test_tensor.transpose(1, 2)
    # 载入模型和参数
    model = TCN(config.input_size, config.output_size, config.num_channels, config.kernel_size, config.dropout)
    model.load_state_dict(torch.load(config.save_path))
    model.to(torch.device("cpu"))
    # 消除dropout层的影响
    model = model.eval()
    # 将数据输入模型进行预测并反归一化
    y_test_pred = scaler.inverse_transform((model(test_tensor).detach().numpy().reshape(-1, 1)))
    ###开始输出预测的效果
    np.set_printoptions(suppress=True)
    print('前50天的历史数据:', scaler_model.inverse_transform((test_)))  # 打印出输入的数据
    print('前50天的历史数据输入模型的shape:', test_tensor.shape)
    print('第51天的预测收盘价:', y_test_pred.squeeze(0))
    # label.reshape(1, 1)
    # label = np.array([[label]])、
    # label = torch.tensor(label).view(1, 1)
    # print('第51天的真实收盘价:', scaler.inverse_transform((label)))
    return y_test_pred.squeeze(0)

# predict(df)
