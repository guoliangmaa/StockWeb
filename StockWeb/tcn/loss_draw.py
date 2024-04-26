#loss可视化
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt #绘图
from StockWeb.Config import config #从自定义的Config.py参数文件中插入

# 加载参数
config = config()
# 读取csv中指定列的数据
data = pd.read_csv("StockWeb/tcn/loss.csv")
train_loss = data[['train_loss']]
test_loss = data[['test_loss']]
y1 =np.array(train_loss)#将DataFrame类型转化为numpy数组
y2 = np.array(test_loss)
# 绘图
plt.plot(range(1, config.epochs+1), y1, label="train_loss")
plt.plot(range(1, config.epochs+1), y2, label="test_loss")
plt.xlabel('epochs')
plt.ylabel('loss')
# plt.ylim(0.0002, 0.00060)  # 设置Y轴范围
# plt.ylim(0.0000, 0.0001)
plt.legend()   #显示标签
plt.show()
