# import os
import os.path

import numpy as np
import torch.nn as nn
import torchvision as tv
from torch.autograd import Variable
import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import torch
from random import sample

from Dataset import ConditionalImageDataset

plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文标签
plt.rcParams['axes.unicode_minus'] = False

#考虑加差分隐私避免重构    准确概率  1

dir = 'E:/k/test/data'


noiseSize = 100     # 噪声维度
n_generator_feature = 64        # 生成器feature map数
n_discriminator_feature = 64        # 判别器feature map数
batch_size = 64
d_every = 1     # 每一个batch训练一次discriminator
g_every = 5     # 每五个batch训练一次generator
num_classes = 100

class NetGenerator(nn.Module):
    def __init__(self, num_classes):
        super(NetGenerator,self).__init__()
        self.main = nn.Sequential(      # 神经网络模块将按照在传入构造器的顺序依次被添加到计算图中执行
            nn.ConvTranspose2d(noiseSize + num_classes, n_generator_feature * 8, kernel_size=4, stride=1, padding=0, bias=False),
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

    def forward(self, noises, labels):
        batch_size = noises.size(0)
        noises = noises.view(batch_size, noiseSize, 1, 1).cuda()
        labels = labels.unsqueeze(1).unsqueeze(2).unsqueeze(3).repeat(1, num_classes, 1, 1).cuda()
        inputs = torch.cat((noises, labels), dim=1)
        return self.main(inputs)


class NetDiscriminator(nn.Module):
    def __init__(self, num_classes):
        super(NetDiscriminator,self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(4, n_discriminator_feature, kernel_size=5, stride=3, padding=1, bias=False),
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

    def forward(self, input,labels):
        labels = labels.view(labels.size(0),-1,1,1)
        labels = labels.repeat(1,1,input.size(2),input.size(3))
        input = torch.cat((input,labels),dim = 1)
        return self.main(input).view(-1)

# def load_two_input_model():
#     checkpoint = torch.load(r'H:\code\dome\test\concat_model1.pt', map_location=torch.device('cpu'))
#     model_ = model.fine_tune_model()
#     model_.load_state_dict(checkpoint,False)
#     model_.eval()
#     return model_

def add_laplace_noise(des, sensitivity, epsilon):
    # des = des.astype(float)  # 将 des 数组转换为浮点数类型
    noise = np.random.laplace(loc=0, scale=sensitivity / epsilon, size=des.shape)
    noisy_des = des.detach().numpy() + noise

    return noisy_des

def get_concated_data(rawinputs1, rawinput2, batch_size):
    # 二和一、每次往后迭代两次
    # 前半部分不打乱、后半部分打乱
    # cnt 记录拼接次数、一旦超过一半后
    # 随机挑选后面的数字 6000 3000 3000 1500 1500
    # 3000 if cnt > 1500: random_pic (1500,3000)
    length = int(batch_size)
    inputs = torch.zeros([length, 3, 192, 96], dtype=torch.float32)
    labels = torch.zeros(length, dtype=torch.int64)
    it_times = int(batch_size)
    cnt = 0
    j = 0
    for i in range(it_times):  # 0  2  4
        inputs[j] = torch.cat([rawinputs1[j], rawinput2[j]], 1)
        labels[j] = 0
        j = j + 1
            # 顺序抽取
        #总之，这段代码的目的是生成一个新的数据集，其中每个样本由两个原始样本拼接而成，并根据这两个原始样本的标签情况确定新样本的标签。
    return inputs, labels

def train():

    criterion_identity =nn.CrossEntropyLoss()  # 尝试换不同的损失函数
    similarity_loss = nn.MSELoss()

    # resnet_model = load_two_input_model()
    # resnet_model.eval()  # Set model to evaluation mode
    for i, (image,labels) in tqdm.tqdm(enumerate(dataloader)):       # type((image,_)) = <class 'list'>, len((image,_)) = 2 * 256 * 3 * 96 * 96
        real_image = Variable(image)
        real_image = real_image.cuda()
        labels = labels.cuda()

        if (i + 1) % d_every == 0:
            optimizer_d.zero_grad()
            output = Discriminator(real_image,labels)      # 尽可能把真图片判为True
            error_d_real = criterion(output, true_labels)
            error_d_real.backward()

            # output = concat_version.all_out
            output = torch.randn(64, 100)
            # 调整维度
            new_tensor = torch.unsqueeze(output, dim=2)
            new_tensor = torch.unsqueeze(new_tensor, dim=3)

            noises.data.copy_(new_tensor)
            fake_img = Generator(noises,labels).detach()       # 根据噪声生成假图
            fake_output = Discriminator(fake_img,labels)       # 尽可能把假图片判为False
            # real_palm_feature = resnet_model(real_image)
            # fake_palm_feature = resnet_model(fake_img)
            # inputs, labels = get_concated_data(real_image, fake_img, batch_size)
            # out = resnet_model(inputs)

            # error_d_fake = criterion(fake_output, fake_labels) + criterion_identity(out,labels)
            error_d_fake = criterion(fake_output, fake_labels)
            error_d_fake.backward()
            optimizer_d.step()

        if (i + 1) % g_every == 0:
            optimizer_g.zero_grad()
            noises.data.copy_(torch.randn(batch_size, noiseSize, 1, 1))
            fake_img = Generator(noises,labels)        # 这里没有detach
            fake_output = Discriminator(fake_img,labels)       # 尽可能让Discriminator把假图片判为True
            # error_g = criterion(fake_output, true_labels)
            # inputs, labels = get_concated_data(real_image, fake_img, batch_size)
            # out = resnet_model(inputs)
            error_g = criterion(fake_output, true_labels)
            error_g.backward()
            optimizer_g.step()

def show(num):
    fixed_labels = torch.arange(0, 64)
    print(fixed_labels)
    fix_fake_imags = Generator(fix_noises,fixed_labels)
    fix_fake_imags = fix_fake_imags.data.cpu()[:64] * 0.5 + 0.5

    # x = torch.rand(64, 3, 96, 96)
    fig = plt.figure(1)

    save_path = "E:/k/test/"

    i = 1
    count = 0
    update_dir = os.path.join(save_path, f"{num}")
    if not os.path.exists(update_dir):
        os.makedirs(update_dir)
    for image in fix_fake_imags:
        ax = fig.add_subplot(8, 8, eval('%d' % i))
        # plt.xticks([]), plt.yticks([])  # 去除坐标轴
        plt.axis('off')
        plt.imshow(image.permute(1, 2, 0))
        i += 1
        # img_path = os.path.join(update_dir, f"{count}.bmp")
        # save_image(image, img_path)
        count = count+1

    plt.subplots_adjust(left=None,  # the left side of the subplots of the figure
                        right=None,  # the right side of the subplots of the figure
                        bottom=None,  # the bottom of the subplots of the figure
                        top=None,  # the top of the subplots of the figure
                        wspace=0.05,  # the amount of width reserved for blank space between subplots
                        hspace=0.05)  # the amount of height reserved for white space between subplots)
    plt.suptitle('第%d迭代结果' % num, y=0.91, fontsize=15)
    plt.savefig("images/%dcgan.png" % num)


if __name__ == '__main__':
    transform = tv.transforms.Compose([
        tv.transforms.Resize(96),     # 图片尺寸, transforms.Scale transform is deprecated
        tv.transforms.CenterCrop(96),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))       # 变成[-1,1]的数
    ])

    # dataset = tv.datasets.ImageFolder(dir, transform=transform)
    #
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)   # module 'torch.utils.data' has no attribute 'DataLoder'

    dataset = tv.datasets.ImageFolder(dir, transform=transform)
    dataset = ConditionalImageDataset(
        root_dir='E:/k/test/data/train',
        transform=transform)
    # 创建数据加载器
    dataloader = DataLoader(dataset, batch_size=64, num_workers=0,drop_last=True)

    # 初始化生成器和判别器
    Generator = NetGenerator( num_classes=100)  # 假设生成器接受类别数和特征数作为参数
    Discriminator = NetDiscriminator(num_classes=100)  # 假设判别器也是

    print('数据加载完毕！')
    # Generator = NetGenerator()
    # Discriminator = NetDiscriminator()

    optimizer_g = torch.optim.Adam(Generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
    optimizer_d = torch.optim.Adam(Discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))
    criterion = torch.nn.BCELoss()

    true_labels = Variable(torch.ones(batch_size))     # batch_size
    fake_labels = Variable(torch.zeros(batch_size))
    fix_noises = Variable(torch.randn(batch_size, noiseSize, 1, 1))
    noises = Variable(torch.randn(batch_size, noiseSize, 1, 1))     # 均值为0，方差为1的正态分布

    if torch.cuda.is_available() == True:
        print('Cuda is available!')
        Generator.cuda()
        Discriminator.cuda()
        criterion.cuda()
        true_labels, fake_labels = true_labels.cuda(), fake_labels.cuda()
        fix_noises, noises = fix_noises.cuda(), noises.cuda()


    plot_epoch = [1,5,10,25,50,100,150,200,250,300,350,400,500,800,1000,1500,2000,2500,3000]

    # test()

    for i in range(3000):        # 最大迭代次数
        train()
        print('迭代次数：{}'.format(i))
        if i in plot_epoch:
            show(i)

    #保存模型
    torch.save(Generator.state_dict(),'./save/gan_/generator.pth')
    torch.save(Discriminator.state_dict(),'./save/gan_/discriminator.pth')