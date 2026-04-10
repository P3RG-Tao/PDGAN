#!/usr/bin/env python
# coding: utf-8

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn

from torch.autograd import Variable

from torch.autograd import Variable

# 安装opencv库，用pip install opencv-python 会导致出错，必须指定版本
# pip install opencv-python==3.4.2.16
# pip install opencv-contrib-python==3.4.2.16
# conda install -c menpo opencv #For Anaconda User just this instead of pip

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
def gan_test(output):
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


def get_train_and_test_img_features():
    train_path = "./Palmprint/training/"
    test_path = "./Palmprint/testing/"
    train_dataset = []  # 存储训练集中每张图片的SIFT特征描述向量
    test_dataset = []  # 存储测试集中每张图片的SIFT特征描述向量

    train_img_list = os.listdir(train_path)
    test_img_list = os.listdir(test_path)

    for train_img in train_img_list:
        img = cv2.imread(train_path + train_img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        equalize = cv2.equalizeHist(gray)  # step1.预处理图片，灰度均衡化
        kp_query, des_query = get_sift_features(equalize)  # step2.获取SIFT算法生成的关键点kp和描述符des(特征描述向量)
        gan_test(des_query)
        train_dataset.append(des_query)
    for test_img in test_img_list:
        img = cv2.imread(test_path + test_img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        equalize = cv2.equalizeHist(gray)
        kp_query, des_query = get_sift_features(equalize)
        gan_test(des_query)
        test_dataset.append(des_query)
    return train_dataset, test_dataset

def add_laplace_noise(des, sensitivity, epsilon):
    des = des.astype(float)  # 将 des 数组转换为浮点数类型
    noise = np.random.laplace(loc=0, scale=sensitivity / epsilon, size=des.shape)
    noisy_des = des + noise
    return noisy_des

def get_sift_features(img, dect_type='sift'):
    if dect_type == 'sift':
        sift = cv2.SIFT_create()
    elif dect_type == 'surf':
        sift = cv2.xfeatures2d.SURF_create()
    kp, des = sift.detectAndCompute(img, None)  # kp为关键点，des为描述符
    des = add_laplace_noise(des, 1.0, 1.0)
    return kp, des


def sift_detect_match_num(des_query, des_train, ratio=0.70):
    # step3.使用KNN计算查询图像与训练图像之间匹配的点数目,采用k(k=2)近邻匹配，最近的点距离与次近点距离之比小于阈值ratio就认为是成功匹配。
    bf = cv2.BFMatcher()
    des_query = des_query.astype(np.float32)
    des_train = des_train.astype(np.float32)
    matches = bf.knnMatch(des_query, des_train, k=2)
    match_num = 0
    for first, second in matches:
        if first.distance < ratio * second.distance:
            match_num = match_num + 1
    return match_num


def get_one_palm_match_num(des_query, index, train_dataset, ratio=0.70):
    # 获取查询图像与训练图像中属于每一组的3张图像的匹配点的数量和
    match_num_sum = 0
    for i in range(index, index + 3):
        match_num_sum += sift_detect_match_num(des_query, train_dataset[i], ratio=ratio)
    return match_num_sum


def get_match_result(des_query, train_dataset, ratio=0.70):
    # step4.根据最大的匹配点数量和，确定查询图片的类别
    index = 0
    train_length = len(train_dataset)
    result = np.zeros(train_length // 3, dtype=np.int32)
    while index < train_length:
        result[index // 3] = get_one_palm_match_num(des_query, index, train_dataset, ratio=ratio)
        index += 3
    return result.argmax()


def predict(train_features, test_features, ratio=0.70):
    predict_true = 0
    for i, feature in enumerate(test_features):
        print('Processing image', i + 1,'...')
        # 预测每张测试图片的类别
        category = get_match_result(feature, train_features, ratio=ratio)
        if category == i // 3:
            predict_true += 1
        print('Predict result:', category + 1, 'Groud truth:', i // 3 + 1)
    print('Predict the correct number of pictures:', predict_true, 'Accuracy:', predict_true / len(test_features), 'ratio:', ratio)
    return predict_true / len(test_features)


def show_plot(ratio, acc, name, title):
    # 绘制准确率变化图
    plt.plot(ratio, acc)
    plt.title(title)
    if not os.path.exists('Image_result'):
        os.makedirs('Image_result')
    plt.savefig(os.path.join('Image_result', name))


def main():
    train_sift_features, test_sift_features = get_train_and_test_img_features()  # 存储每张图片的SIFT特征描述向量
    ratio = 0.65
    best_acc = 0
    best_ratio = 0
    ratio_list = []
    acc_list = []
    max_ratio = 0.85
    while ratio <= max_ratio:  # 循环测试具有最高准确率的ratio
        acc = predict(train_sift_features, test_sift_features, ratio)
        acc_list.append(acc)
        ratio_list.append(ratio)
        if acc > best_acc:
            best_acc = acc
            best_ratio = ratio
        ratio += 0.01
    title = 'best ratio:' + str(best_ratio) + " best acc:{:.4f}".format(best_acc)
    plt_name = "SIFT_" + str(max_ratio).split('.')[-1]
    show_plot(ratio_list, acc_list, plt_name, title)
    print(title)


if __name__ == '__main__':
    main()
