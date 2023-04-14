# 导入必要的库和模块
import argparse  # 解析命令行参数的工具包
import random  # 解析命令行参数的工具包
import torch
import torch.nn.functional as F
from datasets.my_datasets2 import My_Datasets  # 自定义数据集类
import torch.utils.data as data  # 图像处理工具包
from torchvision import transforms  # 图像处理工具包
from torch.autograd import Variable  # 自动求导工具包
from nets.DenseNet01 import Define_DenseNet  # 导入自定义的DenseNet框架
import torch.nn as nn   # PyTorch自带的神经网络层函数
import torch.optim as optim  # PyTorch自带的优化器函数
from torchvision.utils import save_image  # 图像保存工具包
import pandas as pd  # 数据处理工具包
from pytorch_msssim import ssim  # 计算结构相似性指标的包

# 解析命令行参数，这部分定义在训练过程中需要使用的一些参数，例如训练数据的路径、批处理大小、学习率等，并通过argparse模块进行命令行解析。获取opt参数对象。
parser = argparse.ArgumentParser()
parser.add_argument('--raw_traindata_root', default='./datasets/EUVP Dataset/Paired/underwater_imagenet/trainA/', help='folder of raw training data')
parser.add_argument('--target_traindata_root', default='./datasets/EUVP Dataset/Paired/underwater_imagenet/trainB/', help='folder of target training data')
parser.add_argument('--batchsize', type=int, default=4, help='batch size')
parser.add_argument('--niter', type=int, default=300, help='the number of epochs to train for')
parser.add_argument('--inputChannelSize', type=int, default=3, help='the channel size of raw image')
parser.add_argument('--inputImageSize', type=int, default=256, help='the size of raw image')
parser.add_argument('--targetChannelSize', type=int, default=3, help='the channel size of target image')
parser.add_argument('--targetImageSize', type=int, default=256, help='the size of target image')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate')

opt = parser.parse_args()  # 将所有参数存储在opt变量
print(opt)

# 设置随机种子random send，保证随机结果可重现。
opt.manualSeed = 101
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
print("random seed:", opt.manualSeed)

# 数据预处理
raw_transforms = transforms.Compose([
    transforms.ToTensor(),  # 转换为张量
])
target_transforms = transforms.Compose([
    transforms.ToTensor(),  # 转换为张量
])

# 使用My_Datasets类来加载数据集，并构造数据加载器train_data_loader。
train_data = My_Datasets(opt.raw_traindata_root, opt.target_traindata_root, raw_transforms, target_transforms)  # 加载自定义数据集
train_data_loader = data.DataLoader(train_data, batch_size=opt.batchsize, num_workers=0, shuffle=True, drop_last=True)  # 加载自定义数据集

# 初始化输入和输出张量 如初始化一些变量和模型，包括raw_image, target_image, target_out和Unet模型。
raw_image = torch.FloatTensor(opt.batchsize, opt.inputChannelSize, opt.inputImageSize, opt.inputImageSize)
target_image = torch.FloatTensor(opt.batchsize, opt.targetChannelSize, opt.targetImageSize, opt.targetImageSize)
target_out = torch.FloatTensor(opt.batchsize, opt.targetChannelSize, opt.targetImageSize, opt.targetImageSize)
raw_image = Variable(raw_image)
target_image = Variable(target_image)
target_out = Variable(target_out)
epoch = 0

# 定义DenseNet模型
net = Define_DenseNet()
# 初始化模型
for m in net.modules():
    if isinstance(m, torch.nn.Conv2d):
        torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, val=0.0)
    elif isinstance(m, torch.nn.BatchNorm2d):
        torch.nn.init.constant_(m.weight, val=1.0)
        torch.nn.init.constant_(m.bias, val=0.0)
    elif isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, val=0.0)

# 定义损失函数criterion_1和优化器optimizer。
criterion_1 = nn.L1Loss()
optimizer = optim.Adam(net.parameters(), lr=opt.lr, betas=(0.9, 0.999), eps=1e-8)
# 将模型放在GPU上训练
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net = net.to(device)

# 定义用于记录训练过程中的损失的列表
train_loss = []
validation_loss = []
train_loss_output = []
validation_loss_output = []

# 训练循环

for epoch in range(opt.niter):
    for i, (raw_image, target_image) in enumerate(train_data_loader):
        # 设置梯度更新开启
        for p in net.parameters():
            p.requires_grad = True
        # 清空梯度
        net.zero_grad()

        # 将数据移动到GPU上
        raw_image = raw_image.to(device)
        target_image = target_image.to(device)
        # 模型前向传播
        target_out = net(raw_image)
        # 计算损失函数
        target_out = F.interpolate(target_out, size=target_image.size()[2:], mode='bilinear')
        loss_1 = criterion_1(target_image, target_out)
        loss_2 = 1 - ssim(target_image, target_out)
        loss = 0.2*loss_1 + 0.8 * loss_2
        # 反向传播计算梯度
        loss.backward()
        # 更新参数
        optimizer.step()
        # 保存当前批次的训练损失
        train_loss.append(loss.data.item())
        # 每隔20个batch打印一次训练损失
        if (i+1) % 20 == 0:
            print('epoch[{}/{}],loss:{:.6f}'.format(epoch, opt.niter, loss.data.item()))
    # 在第一个epoch结束后保存真实图像和原始图像
    if epoch == 0:
        save_image(target_image, './img/target_image.png')
        save_image(raw_image, './img/raw_image.png')  # 保存去雾后的图像到文件中

    # 计算每个epoch的平均训练损失并将其存储在train_loss_output列表中
    train_loss_eachepoch = sum(train_loss)/len(train_loss)
    train_loss_output.append(train_loss_eachepoch)

    # 保存当前epoch的预测结果
    target_outs = target_out.data
    save_image(target_outs, './img/target_out-{}.png'.format(epoch+1))

# 将每个epoch的平均训练损失存储到CSV文件中
train_loss_output_file = pd.DataFrame(data=train_loss_output)

torch.save(net.state_dict(), './Densenet_Denseblock_net.pth')  # 将训练好的DenseNet模型保存到本地


