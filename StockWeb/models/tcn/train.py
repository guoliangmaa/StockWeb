# #训练的模板函数，做其他项目也可以用这个模板改
# import torch
# from Config import config
# from tqdm import tqdm #进度条
# import numpy as np
# config = config()
#
# def fit(epoch, model, loss_function, optimizer, train_loader, test_loader, bst_loss):
#     model.train()
#     running_loss = 0
#     train_bar = tqdm(train_loader)  # 形成进度条
#     for data in train_bar:
#         x_train, y_train = data  # 取出数据中的X和Y
#         optimizer.zero_grad() #梯度初始化为零
#         y_train_pred = model(x_train) #前向传播求出预测的值
#         loss = loss_function(y_train_pred, y_train.reshape(-1, 1)) #计算每个batch的loss
#         loss.backward()# 反向传播求梯度
#         optimizer.step()# 更新所有参数
#         running_loss += loss.item() #一个epochs里的每次的batchs的loss加起来
#         train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1, config.epochs, loss)
#     epoch_loss = running_loss / len(train_loader.dataset) #一个epochs训练完后，把累加的loss除以batch的数量，得到这个epochs的损失
#     # 模型验证
#     model.eval()
#     test_running_loss = 0
#     with torch.no_grad(): #验证过程不需要计算梯度
#         test_bar = tqdm(test_loader)# 形成进度条
#         for data in test_bar:
#             x_test, y_test = data #取出数据中的X和Y
#             y_test_pred = model(x_test) #求出预测的值
#             test_loss = loss_function(y_test_pred, y_test.reshape(-1, 1))#计算每个batch的loss
#             test_running_loss += test_loss.item()
#     epoch_test_loss = test_running_loss / len(test_loader.dataset) #一个epochs训练完后，把累加的loss除以batch的数量，得到这个epochs的损失
#
#     if epoch_test_loss < bst_loss:  #保存验证集最优模型
#         bst_loss = epoch_test_loss
#         torch.save(model.state_dict(), config.save_path)
#
#     return epoch_loss, epoch_test_loss #输出每个epoch的loss用来绘图

#上面的模板不能改变训练的step,下面这个模板可以

import torch
# from .Config import config
from tqdm import tqdm  # 进度条
import numpy as np

# config = config()


def fit(epoch, model, loss_function, optimizer, train_loader, test_loader, bst_loss, config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    model.train()
    running_loss = 0
    train_bar = tqdm(train_loader)  # 形成进度条

    train_steps = len(train_loader) # 计算每个 epoch 中的训练步数，改成train_steps=1的话就是每个epoch只训练一次样本
    # len(train_loader)
    for step, data in enumerate(train_bar):
        if step >= train_steps:
            break

        x_train, y_train = data  # 取出数据中的X和Y
        x_train, y_train = x_train.to(device), y_train.to(device)  # 将数据移动到GPU上
        optimizer.zero_grad()  # 梯度初始化为零
        y_train_pred = model(x_train)  # 前向传播求出预测的值
        loss = loss_function(y_train_pred, y_train.reshape(-1, 1))  # 计算每个batch的loss
        loss.backward()  # 反向传播求梯度
        optimizer.step()  # 更新所有参数
        running_loss += loss.item()  # 一个epochs里的每次的batchs的loss加起来
        train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1, config.epochs, loss)
    # epoch_loss = running_loss / len(train_loader.dataset)
    epoch_loss = running_loss / (len(train_loader.dataset) * (train_steps / len(train_loader)))  # 一个epochs训练完后，把累加的loss除以实际训练样本数量，得到这个epochs的损失

    # 模型验证
    model.eval()
    test_running_loss = 0
    with torch.no_grad():  # 验证过程不需要计算梯度
        test_bar = tqdm(test_loader)  # 形成进度条
        for data in test_bar:
            x_test, y_test = data  # 取出数据中的X和Y
            x_test, y_test = x_test.to(device), y_test.to(device)  # 将数据移动到GPU上
            y_test_pred = model(x_test)  # 求出预测的值
            test_loss = loss_function(y_test_pred, y_test.reshape(-1, 1))  # 计算每个batch的loss
            test_running_loss += test_loss.item()
            test_bar.desc = "test epoch[{}/{}] test running loss:{:.3f}".format(epoch + 1, config.epochs, test_running_loss)
    epoch_test_loss = test_running_loss / len(test_loader.dataset)  # 一个epochs训练完后，把累加的loss除以batch的数量，得到这个epochs的损失
    # epoch_loss = running_loss / (len(train_loader.dataset) * (train_steps / len(train_loader)))
    if epoch_test_loss < bst_loss:  # 保存验证集最优模型
        bst_loss = epoch_test_loss
        torch.save(model.state_dict(), config.save_path)
    print('\nepoch:', epoch, ', train_loss:', round(epoch_loss, 8),
          ', test_loss:', round(epoch_test_loss, 8),
          )
    return epoch_loss, epoch_test_loss  # 输出每个epoch的loss用来绘图
