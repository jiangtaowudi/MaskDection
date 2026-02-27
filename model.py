import torch
from torch import nn
from torchsummary import summary

# 定义残差快类
class Residual(nn.Module):
    def __init__(self, in_channels, out_channels,use_conv1=False,strides=1):
        super(Residual, self).__init__()
        self.ReLU = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if use_conv1:
            self.conv3 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=strides)
        else:
            self.conv3 = None

    def forward(self, x):
        y = self.ReLU(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        if self.conv3:
            x = self.conv3(x)
        y = self.ReLU(y + x)
        return y

# 定义ResNet18
class ResNet18(nn.Module):
    def __init__(self, Residual):
        super(ResNet18, self).__init__()
        self.b1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.b2 = nn.Sequential(
            Residual(64, 64, use_conv1=False,strides=1),
            Residual(64, 64, use_conv1=False,strides=1)
        )

        self.b3 = nn.Sequential(
            Residual(64, 128, use_conv1=True, strides=2),
            Residual(128, 128, use_conv1=True, strides=1)
        )

        self.b4 = nn.Sequential(
            Residual(128, 256, use_conv1=True, strides=2),
            Residual(256, 256, use_conv1=False, strides=1)
        )

        self.b5 = nn.Sequential(
            Residual(256, 512, use_conv1=True, strides=2),
            Residual(512, 512, use_conv1=False, strides=1)
        )

        self.b6 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(512, 10)
        )

        # # 权重初始化
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, nonlinearity='relu', mode='fan_out')
        #         if m.bias is not None:
        #             nn.init.constant_(m.bias, 0)
        #     elif isinstance(m, nn.Linear):
        #         nn.init.normal_(m.weight, 0, 0.01)

    def forward(self,x):
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        x = self.b5(x)
        x = self.b6(x)
        return x

if __name__ == "__main__":
    # 选择处理器
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 模型实例化
    model = ResNet18(Residual).to(device)

    # 打印模型信息
    print(summary(model, (3,224,224)))


