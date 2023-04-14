import torch.nn as nn
import torch

class Dense_Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Dense_Block, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.conv4 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.relu(self.bn3(self.conv3(out)))
        out = self.relu(self.bn4(self.conv4(out)))
        out = torch.cat([x, out], 1)

        return out


class Transition_Layer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Transition_Layer, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)  # 修改池化层参数从kernel_size=4变为2 保留更多原始图像信息

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.avg_pool(out)

        return out


class Define_DenseNet(nn.Module):
    def __init__(self, growth_rate=32, num_layers=4):
        super(Define_DenseNet, self).__init__()

        self.init_conv = nn.Conv2d(3, growth_rate * 4, kernel_size=3, padding=1, bias=False)  # 修改初始化卷积层输出通道数growth_rate * 2改为growth_rate * 4，以提高模型的表达能力和特征提取能力
        self.init_bn = nn.BatchNorm2d(growth_rate * 4)
        self.relu = nn.ReLU(inplace=True)

        in_channels = growth_rate * 4
        for i in range(num_layers):
            dense_block = Dense_Block(in_channels, growth_rate)
            in_channels += growth_rate
            setattr(self, 'dense_block{}'.format(i), dense_block)

            if i != num_layers - 1:
                transition_layer = Transition_Layer(in_channels, in_channels // 2)
                in_channels = in_channels // 2
                setattr(self, 'transition_layer{}'.format(i), transition_layer)

        self.final_bn = nn.BatchNorm2d(in_channels)
        self.final_conv = nn.Conv2d(in_channels, 3, kernel_size=3, padding=1, bias=False)  # 修改输出通道数为3，生成RGB彩色图像
        self.sigmoid = nn.Sigmoid()
        # 添加一些预处理和后处理层，例如去雾、颜色平衡等
        self.preprocess = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(3, 3, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
        )
        self.postprocess = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(3, 3, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(3, 3, kernel_size=5, padding=2),
        )

    def forward(self, x):
        x = self.preprocess(x)  # 添加预处理层
        out = self.relu(self.init_bn(self.init_conv(x)))
        for i in range(4):
            dense_block = getattr(self, 'dense_block{}'.format(i))
            out = dense_block(out)

            if i != 3:
                transition_layer = getattr(self, 'transition_layer{}'.format(i))
                out = transition_layer(out)

        out = self.final_bn(out)
        out = self.relu(out)
        out = self.final_conv(out)
        out = self.sigmoid(out)
        out = self.postprocess(out)  # 添加后处理层
        out = nn.functional.interpolate(out, (256, 256))  # 将输出大小调整为(256, 256)

        return out
