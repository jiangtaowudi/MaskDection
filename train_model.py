import copy
import time
from torchvision.datasets import ImageFolder
from torchvision import transforms
import torch.utils.data as Data
import numpy as np
import matplotlib.pyplot as plt
from model import Residual, ResNet18
import torch
import torch.nn as nn
import pandas as pd


# 定义数据处理函数
def train_val_data_process():
    """定义数据加载函数"""

    # 获取训练数据集路径
    ROOT_train = r'data/train'

    # 数据归一化处理
    normalize = transforms.Normalize([0.173, 0.151, 0.143], [0.074, 0.062, 0.059])

    # 数据处理
    transforms_train = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), normalize])

    # 加载数据
    datasets = ImageFolder(ROOT_train, transforms_train)

    # 划分数据集
    train_data, val_data = Data.random_split(datasets,[round(0.8*len(datasets)),round(0.2*len(datasets))])

    # 封装训练集
    train_loder = Data.DataLoader(dataset=train_data,
                                  batch_size=32,
                                  shuffle=True,
                                  num_workers=2)

    # 封装验证集
    val_loder = Data.DataLoader(dataset=val_data,
                                batch_size=32,
                                shuffle=True,
                                num_workers=2)

    return train_loder, val_loder


# 定义训练模型函数
def train_model_process(model, train_loder, val_loder, num_epochs):
    """训练模型函数"""
    # 选择训练设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 设置优化器，选择Adam优化，并设置学习率为0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # 分类使用交叉损失函数，回归使用均方损失函数
    criterion = nn.CrossEntropyLoss()
    # 将模型导入到训练设备中
    model = model.to(device)
    # 复制当前模型的参数
    best_model_wts = copy.deepcopy(model.state_dict())

    # 初始化参数
    # 最高精度
    best_acc = 0.0
    # 训练集损失列表
    train_loss_all = []
    # 验证集损失列表
    val_loss_all = []
    # 训练集准确度列表
    train_acc_all = []
    # 验证集准确度列表
    val_acc_all = []
    # 训练时间
    since = time.time()

    # 遍历训练轮次
    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs-1))   # 打印每一轮的训练轮次
        print("-"*10)

        # 每个轮次刷新一次初始化参数
        # 训练集损失参数
        train_loss = 0.0
        # 训练集准确度
        train_corrects = 0
        # 验证集损失参数
        val_loss = 0.0
        # 验证集准确度
        val_corrects = 0
        # 训练集样本数量
        train_num = 0
        # 验证集样本数量
        val_num = 0

        # 对每一个mini-batch训练和计算
        for step, (b_x, b_y) in enumerate(train_loder):
            b_x = b_x.to(device)  # 将b_x放入到设备中去，进行实例化
            b_y = b_y.to(device)  # 将b_y放入到设备中去，进行实例化
            # 设置为训练模式
            model.train()

            # 前向传播过程，输入为一个batch，输出为一个batch中对应的预测
            output = model(b_x)   # model定义了forward函数，输入b_x，返回前向传播的x
            # 计算输出概率并找到这一行中的标签最大值
            pre_lab = torch.argmax(output, dim=1)
            # 计算前向传播的交叉损失函数
            loss = criterion(output, b_y)  # 计算出来的损失值为平均值

            # 将梯度初始化为0，避免上一轮对本轮产生影响
            optimizer.zero_grad()
            # 反向传播计算
            loss.backward()
            # 根据网络反向传播的梯度信息来更新网络的参数，以起到降低loss函数计算值的作用
            optimizer.step()
            # 对损失函数进行累加
            train_loss += loss.item() * b_x.size(0)
            # loss.item()表示损失函数的平均值
            # b_x.size(0)获取当前批次中样本的数量。在PyTorch中，size(0) 指的是张量在指定维度上的大小，对于批次数据来说，这个维度通常代表样本数量。

            # 预测正确的话，则训练集准确度+1
            train_corrects += torch.sum(pre_lab == b_y.data)
            # 计算当前用于预测的训练集样本数量
            train_num += b_x.size(0)
        # 验证集loss计算 验证集不需要反向传播
        for step, (b_x, b_y) in enumerate(val_loder):
            b_x = b_x.to(device)
            b_y = b_y.to(device)
            # 设置为验证模式
            model.eval()
            # 前向传播过程，输入为一个batch，输出为一个batch中对应的预测
            output = model(b_x)
            # 获取每一行的最大值对应的下标
            pre_lab = torch.argmax(output, dim=1)
            # 计算交叉损失函数
            loss = criterion(output, b_y)

            # 对损失函数进行累加
            val_loss += loss.item() * b_x.size(0)
            # 如果预测正确，则验证集准确度＋1
            val_corrects += torch.sum(pre_lab == b_y.data)
            # 计算验证集样本数量
            val_num += b_x.size(0)

        # 计算并保存每一次迭代的loss值和准确率
        # 计算并保存训练集的loss值(训练集的损失参数除以样本数量)
        train_loss_all.append(train_loss / train_num)
        # 计算并保存训练集的准确度
        train_acc_all.append(train_corrects.double().item() / train_num)

        # 计算并保存验证集的loss值
        val_loss_all.append(val_loss / val_num)
        # 计算并保存验证集的准确度
        val_acc_all.append(val_corrects.double().item() / val_num)
        print("{} train loss:{:.4f} train acc: {:.4f}".format(epoch, train_loss_all[-1], train_acc_all[-1]))
        print("{} val loss:{:.4f} val acc: {:.4f}".format(epoch, val_loss_all[-1], val_acc_all[-1]))

        # 通过if语句找到最优准确率参数并保存最优参数
        if train_acc_all[-1] > best_acc:
            best_acc = train_acc_all[-1]
            # 保存最优参数
            best_model_wts = copy.deepcopy(model.state_dict())

        # 计算训练时长
        time_use = time.time() - since
        print("训练与验证所耗时间：{:.0f}m{:.0f}s".format(time_use // 60, time_use % 60))

    # 保存最优模型
    torch.save(best_model_wts, 'C:/Users/jiang/Desktop/mask_categorize/best_model.pth')

    # 通过pandas进行数据处理,搭建模型框架
    train_process = pd.DataFrame(data={"epoch":range(num_epochs),
                                       "train_loss_data":train_loss_all,
                                       "val_loss_data":val_loss_all,
                                       "train_acc_data":train_acc_all,
                                       "val_acc_data":val_acc_all,})
    return train_process


def matplot_process(train_process):
    # 显示每一次迭代后的训练集和验证集的损失函数和准确率
    # 第一张图用来显示loss值
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)   # 创建1行2列的子图，并显示第一张子图
    plt.plot(train_process['epoch'],train_process.train_loss_data,'ro-',label="Train loss")
    plt.plot(train_process["epoch"],train_process.val_loss_data,'bs-',label="Val loss")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    # 第二张图用于显示准确度
    plt.subplot(1,2,2)
    plt.plot(train_process["epoch"],train_process.train_acc_data,'ro-',label="Train acc")
    plt.plot(train_process["epoch"],train_process.val_acc_data,'bs-',label="Val acc")
    plt.xlabel("Epoch")
    plt.ylabel("Acc")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    ResNet18 = ResNet18(Residual)   # 加载LeNet模型
    # 加载数据集
    train_data, val_data = train_val_data_process()
    # 模型训练
    train_process = train_model_process(ResNet18, train_data, val_data, num_epochs=30)
    # 结果展示
    matplot_process(train_process)
