from torchvision.datasets import ImageFolder
from torchvision import transforms  # 导入转换器
import torch.utils.data as Data
import numpy as np
import matplotlib.pyplot as plt
import torch

# 获取数据路径
ROOT_train = r'data/train'

# 数据归一化处理
normalize = transforms.Normalize([0.173, 0.151, 0.143], [0.074, 0.062, 0.059])

# 数据处理
transforms_train = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), normalize])

# 加载数据
datasets = ImageFolder(ROOT_train, transforms_train)

# 封装训练集
train_loader = Data.DataLoader(dataset=datasets,
                               batch_size=64,
                               num_workers=0,
                               shuffle=True)

# 获得一个Batch的数据
for step, (b_x, b_y) in enumerate(train_loader):
    if step > 0:
        break

batch_x = b_x.squeeze().numpy()   # 将四维张量移除第1维，并转换成Numpy数组
batch_y = b_y.numpy()   # 将张量转换成Numpy数组
batch_label = datasets.classes  # 获取数据的label
print(batch_x.shape)
print("The size of batch in train data:", batch_x.shape)  # 每个mini-batch的维度是64*224*224

# 反归一化参数
mean = np.array([0.173, 0.151, 0.143])
std = np.array([0.074, 0.062, 0.059])

# 展示数据集
plt.figure(figsize=(12, 5))
# 遍历第一簇的64张图像
for ii in range(len(batch_y)):
    plt.subplot(4, 16, ii+1)  # 由于索引从1起始，而非0，所以在循环中我们常用ii + 1来正确设置子图位置
    # 获取单个图像并反归一化
    img = batch_x[ii].copy()
    img = img * std[:, None, None] + mean[:, None, None]
    img = np.clip(img, 0, 1)  # 确保像素值在0-1之间
    # 转换通道顺序为HWC
    img = np.transpose(img, (1, 2, 0))
    plt.imshow(img)   # 显示每一张图片，并设置每一张图片为灰度图
    plt.title(batch_label[batch_y[ii]], size=10)    # 展示每一张图片的标签
    plt.axis('off')     # 不显示x轴的尺寸
    plt.subplots_adjust(wspace=0.06)
plt.show()



