import os

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn

from torch.autograd import Variable

plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
plt.rcParams['axes.unicode_minus'] = False
noiseSize = 1000     # 噪声维度
n_generator_feature = 64        # 生成器feature map数
n_discriminator_feature = 64        # 判别器feature map数
batch_size = 50
d_every = 1     # 每一个batch训练一次discriminator
g_every = 5     # 每五个batch训练一次generator

class NetGenerator(nn.Module):
    def __init__(self):
        super(NetGenerator,self).__init__()
        self.main = nn.Sequential(      # 神经网络模块将按照在传入构造器的顺序依次被添加到计算图中执行
            nn.ConvTranspose2d(noiseSize, n_generator_feature * 8, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(n_generator_feature * 8),
            nn.ReLU(True),       # (n_generator_feature * 8) × 4 × 4        (1-1)*1+1*(4-1)+0+1 = 4
            nn.ConvTranspose2d(n_generator_feature * 8, n_generator_feature * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(n_generator_feature * 4),
            nn.ReLU(True),      # (n_generator_feature * 4) × 8 × 8     (4-1)*2-2*1+1*(4-1)+0+1 = 8
            nn.ConvTranspose2d(n_generator_feature * 4, n_generator_feature * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(n_generator_feature * 2),
            nn.ReLU(True),  # (n_generator_feature * 2) × 16 × 16
            nn.ConvTranspose2d(n_generator_feature * 2, n_generator_feature, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(n_generator_feature),
            nn.ReLU(True),      # (n_generator_feature) × 32 × 32
            nn.ConvTranspose2d(n_generator_feature, 3, kernel_size=5, stride=3, padding=1, bias=False),
            nn.Tanh()       # 3 * 96 * 96
        )

    def forward(self, input):
        return self.main(input)


def test(output):
    net = NetGenerator()
    net.cuda()
    net.load_state_dict(torch.load("./generator_.pth"))
    net.eval()

    if not os.path.exists('output/A'):
        os.makedirs('output/A')
    if not os.path.exists('output/B'):
        os.makedirs('output/B')
    optimizer_g = torch.optim.Adam(NetGenerator().parameters(), lr=2e-4, betas=(0.5, 0.999))
    optimizer_g.zero_grad()
    noises = Variable(torch.randn(batch_size, noiseSize, 1, 1)).cuda()
    # 调整维度
    new_tensor = torch.unsqueeze(output, dim=2)
    new_tensor = torch.unsqueeze(new_tensor, dim=3)
    noises.data.copy_(new_tensor)
    fake_imgs = net(noises)  # 这里没有detach
    fix_fake_imags = fake_imgs.data.cpu()[:64] * 0.5 + 0.5
    fig = plt.figure(1)
    i = 1
    for image in fix_fake_imags:
        ax = fig.add_subplot(8, 8, eval('%d' % i))
        # plt.xticks([]), plt.yticks([])  # 去除坐标轴
        plt.axis('off')
        plt.imshow(image.permute(1, 2, 0))

        output_dir="D:/kyw/Palmprint_Recognition-master/out/gan__/"
        filename = os.path.join(output_dir, f"image_{i}.png")
        # 保存图像
        plt.imsave(filename, image.permute(1, 2, 0).numpy())
        i += 1
    print("jieshu")
    num=1
    plt.subplots_adjust(left=None,  # the left side of the subplots of the figure
                        right=None,  # the right side of the subplots of the figure
                        bottom=None,  # the bottom of the subplots of the figure
                        top=None,  # the top of the subplots of the figure
                        wspace=0.05,  # the amount of width reserved for blank space between subplots
                        hspace=0.05)  # the amount of height reserved for white space between subplots)
    plt.suptitle('第%d迭代结果' % num, y=0.91, fontsize=15)
    plt.savefig("images/%dc.png" % num)

# 加载预训练的ResNet-18模型
resnet = models.resnet18(pretrained=True)

# 设置模型为评估模式
resnet.eval()

# 定义一个预处理函数，将输入图像转换为模型所需的格式
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 加载并预处理输入图像
input_image = Image.open("D:/dome/test/k/train/0/1.jpg")
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0)

# 使用模型获取特征向量
with torch.no_grad():
    features = resnet(input_batch)

# 获取特征向量
feature_vector = features.squeeze().numpy()
print(feature_vector)

# 定义差分隐私参数
epsilon = 1.0

# 生成拉普拉斯噪声
noise = np.random.laplace(0, 1/epsilon, features.shape)

# 加上拉普拉斯噪声
noisy_feature_vector = features + noise
print(noisy_feature_vector.shape)
# 调整维度
new_tensor = torch.unsqueeze(noisy_feature_vector, dim=2)
new_tensor = torch.unsqueeze(new_tensor, dim=3)
print(new_tensor.shape)
fix_noises = Variable(torch.randn(1, 1000, 1, 1))
print(fix_noises.shape)
test(new_tensor)

