import os
import cv2
import numpy as np

# 设置文件夹路径和模糊参数
source_folder = 'E:/k/test/ROI_image/train/20'  # 替换为你的图片文件夹路径
blurred_folder = 'E:/k/test/deal_mid/21'    # 模糊后图片保存的新文件夹
kernel_size = (15, 15)
sigma_x = 30

# 确保模糊后图片的文件夹存在
if not os.path.exists(blurred_folder):
    os.makedirs(blurred_folder)


# 获取文件夹中所有图片的文件名
image_files = [f for f in os.listdir(source_folder) if f.endswith(('.bmp'))]

# 遍历图片文件并应用高斯模糊
for image_file in image_files[:10]:  # 只处理前10张图片
    # 构建原始图片完整路径
    source_path = os.path.join(source_folder, image_file)

    # 读取图片
    img = cv2.imread(source_path)

    # 检查图片是否读取成功
    if img is not None:
        # # 应用高斯模糊
        # blurred_img = cv2.GaussianBlur(img, kernel_size, sigma_x)

        # blurred_img = cv2.blur(img, (16, 16))  # 假设使用5x5的核大小

        # 为ROI添加高斯噪声
        # 注意：OpenCV的addWeighted函数不是用来添加噪声的，我们需要使用numpy的random.normal
        mean = 0  # 高斯噪声的均值
        sigma = 20  # 标准差，你可以设置为9或49来测试效果
        roi = img
        noise = np.random.normal(mean, sigma, roi.shape)
        noisy_roi = roi.astype(np.float32) + noise

        # 限制像素值在0-255范围内
        noisy_roi = np.clip(noisy_roi, 0, 255).astype(np.uint8)


        # 构建模糊后图片的保存路径（不包括文件扩展名）
        base_name, ext = os.path.splitext(image_file)
        blurred_path = os.path.join(blurred_folder, base_name + '_blurred' + ext)

        # 保存模糊后的图片
        cv2.imwrite(blurred_path, noisy_roi)
    else:
        print(f"Error: Unable to read image {source_path}")

print("Gaussian blur applied to 10 images and saved to new folder.")

# # 读取图片
# image1 = cv2.imread('00001.bmp')
# roi = image1.copy()
# # 定义要打马赛克的矩形区域（左上角坐标和宽度、高度）
# x, y, w, h = 100, 100, 200, 200  # 示例值，你需要根据实际需要调整
#
# # 获取马赛克块的大小
# block_size = 10
# # 对ROI进行马赛克处理
# for i in range(0, h, block_size):
#     for j in range(0, w, block_size):
#         # 确保不会超出边界
#         i_end = min(i + block_size, h)
#         j_end = min(j + block_size, w)
#
#         # 截取小块并应用平均滤波（马赛克效果）
#         block = roi
#         block_blurred = cv2.blur(block, (block_size, block_size))
#
#         # 将模糊后的小块放回原位置
#         roi = block_blurred







# roi = cv2.imread('00001.bmp')
# roi = cv2.GaussianBlur(roi, (15, 15), 30)
#
# # 显示输出
# cv2.imshow('Blur Face', roi)
# cv2.waitKey(0)