import torch

import torch.nn as nn

import torchvision.models as models

noiseSize = 100     # 噪声维度
n_generator_feature = 64        # 生成器feature map数
n_discriminator_feature = 64        # 判别器feature map数
batch_size = 50
d_every = 1     # 每一个batch训练一次discriminator
g_every = 5     # 每五个batch训练一次generator


def fine_tune_model(use_gpu=False):
    model_feature = models.resnet18(pretrained=True)
    # 记录全连接层的特征输入数
    num_features = model_feature.fc.in_features
    # 将最后一层的全连接层修改为我们定义的层
    model_feature.fc = nn.Linear(num_features, 2)
    if use_gpu:
        model_feature = model_feature.cuda()
    return model_feature


def get_two_input_net():
    model_feature = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    net = two_input_net(model_feature)
    return net


class two_input_net(nn.Module):
    def __init__(self, model):
        super(two_input_net, self).__init__()
        self.conv = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.resnet_layer = nn.Sequential(*list(model.children())[1:-1])
        # self.fc = nn.Linear(512)

    def forward(self, input1, input2):
        x1 = self.conv(input1)
        x2 = self.conv(input2)
        x1 = self.resnet_layer(x1)  # 50,512,7,7
        x2 = self.resnet_layer(x2)  # 50,512,7,7
        # x1 = self.linear(x1)
        # x2 = self.linear(x2)
        # torch.cat([x1, x2], )
        x1 = torch.squeeze(x1)  # 50, 512 , 1 ,1
        x2 = torch.squeeze(x2)  # 50, 512 , 1 ,1
        # print(x1.shape)
        # cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        # output = cos(x1, x2)  # 50
        # print(output)
        # print(output.shape)
        return x1, x2  # 50

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


class NetDiscriminator(nn.Module):
    def __init__(self):
        super(NetDiscriminator,self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, n_discriminator_feature, kernel_size=5, stride=3, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),        # n_discriminator_feature * 32 * 32
            nn.Conv2d(n_discriminator_feature, n_discriminator_feature * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(n_discriminator_feature * 2),
            nn.LeakyReLU(0.2, inplace=True),         # (n_discriminator_feature*2) * 16 * 16
            nn.Conv2d(n_discriminator_feature * 2, n_discriminator_feature * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(n_discriminator_feature * 4),
            nn.LeakyReLU(0.2, inplace=True),  # (n_discriminator_feature*4) * 8 * 8
            nn.Conv2d(n_discriminator_feature * 4, n_discriminator_feature * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(n_discriminator_feature * 8),
            nn.LeakyReLU(0.2, inplace=True),  # (n_discriminator_feature*8) * 4 * 4
            nn.Conv2d(n_discriminator_feature * 8, 1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid()        # 输出一个概率
        )

    def forward(self, input):
        return self.main(input).view(-1)


if __name__ == '__main__':
    model_feature = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    print(model_feature)
    model = two_input_net(model_feature)
    #print(model)
