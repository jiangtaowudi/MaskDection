import torch
from torchvision.datasets import ImageFolder
from torchvision import transforms
import torch.utils.data as Data
from model import Residual,ResNet18
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# 测试数据集加载函数
def test_data_process():

    # 获取数据路径
    root_test = r'data/test'

    # 进行归一化处理
    normalize = transforms.Normalize([0.173, 0.151, 0.143], [0.074, 0.062, 0.059])

    # 数据格式处理
    trans_test = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), normalize])

    # 加载数据
    test_data = ImageFolder(root_test,trans_test)

    # 封装测试数据集
    test_data_loader = Data.DataLoader(dataset=test_data,
                                       batch_size=1,
                                       shuffle=True,
                                       num_workers=0)

    return test_data_loader

# 定义测试集处理函数
def test_model_process(model,test_data_loader):
    # 选择测试设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # 将模型导入到测试设备中
    model = model.to(device)

    # 设置训练参数
    test_acc = 0.0  # 测试的准确度
    test_num = 0    # 测试的样本数

    # 不考虑优化器，只进行前向传播，不考虑反向传播和梯度计算
    with torch.no_grad():
        for test_x, test_y in test_data_loader:
            # 将应用值导入到模型中
            test_x = test_x.to(device)
            # 将对应的标签导入到模型中
            test_y = test_y.to(device)
            # 设备为验证模式
            model.eval()
            # 进行前向传播，得到结果值
            output = model(test_x)
            # 查找每一行中最大值的索引
            pre_lab = torch.argmax(output, dim=1)
            # 如果预测正确，则准确值加1
            test_acc += torch.sum(pre_lab == test_y.data)
            # 将样本数量相加
            test_num += test_x.size(0)

        # 计算测试的正确率
        test_corrects = test_acc.double().item() / test_num
        print("测试的正确率为：",test_corrects)

# 进行函数测试
if __name__ == "__main__":
    # 导入模型
    model = ResNet18(Residual)
    model.load_state_dict(torch.load('best_model.pth'))
    # # 导入数据集
    # test_data_loader = test_data_process()
    # # 进行模型测试
    # test_model_process(model, test_data_loader)

# 模型推理
    # 设置支持中文字符的字体
    plt.rcParams['font.family'] = 'SimHei'  # 'SimHei' 是常用的中文字体
    # 重建标签列表
    classes = ['戴口罩', '不戴口罩']

    # 读取图片
    image = Image.open('7573.jpg_wh860.jpg')

    # 进行归一化处理
    normalize = transforms.Normalize([0.173, 0.151, 0.143], [0.074, 0.062, 0.059])

    # 数据格式处理
    trans_test = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), normalize])

    # 图片数据处理
    image = trans_test(image)

    # 添加批次维度
    image = image.unsqueeze(0)

    # 选择推理处理器
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 模型实例化
    model = model.to(device)

    # 开始推理
    with torch.no_grad():
        # 图像实例化
        image = image.to(device)
        # 设置模型为验证模式
        model.eval()

        # 将数据放入模型得出推理结果
        output = model(image)

        # 获取预测结果
        pre_lab = torch.argmax(output, dim=1)

        # 将预测结果的下标转换为数值形式
        result = pre_lab.item()

        # 打印模型推理结果
        print("预测值：", classes[result])

        # 反归一化参数
        mean = np.array([0.173, 0.151, 0.143])
        std = np.array([0.074, 0.062, 0.059])

        # 显示预测图片
        batch_x = image.squeeze().cpu().numpy()  # 将四维张量移除第1维，并转换成Numpy数组
        img = batch_x.copy()
        img = img * std[:, None, None] + mean[:, None, None]
        img = np.clip(img, 0, 1)  # 确保像素值在0-1之间
        # 转换通道顺序为HWC
        img = np.transpose(img, (1, 2, 0))
        plt.imshow(img)  # 显示每一张图片
        plt.title(classes[result], size=10)  # 展示每一张图片的标签
        plt.axis('off')  # 不显示x轴的尺寸
        plt.subplots_adjust(wspace=0.06)
    plt.show()



