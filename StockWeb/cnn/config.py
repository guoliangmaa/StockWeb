
class Config:
    data_path = 'StockWeb/cnn/sh300_test.csv'
    timestep = 50  # 滑窗大小
    batch_size = 32  # 训练批次
    input_size = 6  # 输入的维度，特征数量
    output_size = 1  # 预测未来一天的股票收盘价，最终输出层大小为1
    num_channels = [16, 32, 64]
    kernel_size = 5
    dropout = 0
    epochs = 100  # 训练次数
    model_name = 'TCN'  # 模型名称
    # save_path = './{}.pth'.format(model_name)  # 最优模型保存路径
    save_path = 'StockWeb/tcn/{}.pth'.format(model_name)  # 最优模型保存路径
    learning_rate = 0.0001  # 学习率
    show_figure = False  # 是否显示结果图像
    stock_code = '000001'  # 股票代码
    length = 365  # 看多少天以前的数据