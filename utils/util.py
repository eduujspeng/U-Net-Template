# -*- coding:utf-8 -*-
# author:peng
# Date：2024/1/15 10:13
import os
import random

import albumentations as A
import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt

from model.UNet import UNet
from utils.config import CFG


def set_seed(seed=42):
    """
    设置整个笔记本的种子，以便每次运行时结果都相同。
    这是为了保证可重复性。
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # 当在CuDNN后端运行时，必须设置两个进一步的选项
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 为哈希种子设置一个固定值
    os.environ['PYTHONHASHSEED'] = str(seed)

    print('> 种子设置完成')


# 定义函数 load_img，用于加载图像并进行预处理
def load_img(path):
    # 使用 OpenCV 读取图像，保留原始 alpha 通道信息
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    # 将灰度图像转换为 RGB 彩色图像
    # img = np.tile(img[..., None], [1, 1, 3])
    # 将图像数据类型转换为 float32，原始数据类型为 uint16
    img = img.astype('float32')
    # 将图像像素值缩放到 [0, 1] 的范围
    mx = np.max(img)
    if mx:
        img /= mx
    # 返回预处理后的图像
    return img


# 定义函数 load_msk，用于加载掩码并进行预处理
def load_msk(path):
    # 使用 NumPy 加载掩码数据，并将数据类型转换为 float32
    msk_gif = cv2.VideoCapture(path)
    ret, msk = msk_gif.read()
    msk = msk.astype('float32')

    # 将掩码像素值缩放到 [0, 1] 的范围
    msk /= 255.0

    # 返回预处理后的掩码
    return msk


# 定义函数 show_img，用于显示图像及其对应的掩码
def show_img(img, mask=None):
    # 设置图像和掩码的显示样式
    plt.figure(figsize=(12, 12))
    img = img[:, :, ::-1]
    plt.imshow(img, cmap='bone')

    # 如果提供了掩码，则以透明度 0.5 的形式叠加显示
    if mask is not None:
        plt.imshow(mask, alpha=0.5)
        # handles = [Rectangle((0, 0), 1, 1, color=_c) for _c in
        #            [(0.667, 0.0, 0.0), (0.0, 0.667, 0.0), (0.0, 0.0, 0.667)]]
        # labels = ["Large Bowel", "Small Bowel", "Stomach"]
        # plt.legend(handles, labels)

    # 关闭坐标轴显示
    plt.axis('off')
    plt.show()


def plot_batch(imgs, msks, size=3):
    if len(imgs) < size or len(msks) < size:
        raise ValueError("Not enough images or masks to display, reduce the size.")

    plt.figure(figsize=(5 * size, 5))  # Adjusting the figure size based on the number of images
    for idx in range(size):
        # Image
        plt.subplot(2, size, idx + 1)  # Adjusting subplot for images
        img = imgs[idx].permute(1, 2, 0).numpy() * 255.0
        img = img[:, :, ::-1].astype('uint8')
        plt.imshow(img)
        plt.axis('off')  # Turn off axis

        # Mask
        plt.subplot(2, size, idx + 1 + size)  # Adjusting subplot for masks
        msk = msks[idx].permute(1, 2, 0).numpy() * 255.0
        msk = msk.astype('uint8')
        plt.imshow(msk)
        plt.axis('off')  # Turn off axis

    plt.tight_layout()
    plt.show()


def load_model(weight_path=None):
    model = UNet(3, 1)
    if weight_path is not None:
        model.load_state_dict(torch.load(weight_path, map_location=torch.device('cpu')))
        print(weight_path, '加载模型成功')
    else:
        print('未加载模型')
    return model


# 定义数据增强的转换管道
data_transforms = {
    "train": A.Compose([
        # 调整图像大小
        A.Resize(height=CFG.img_size[0], width=CFG.img_size[1], interpolation=cv2.INTER_NEAREST),
        # 水平翻转
        A.HorizontalFlip(p=0.5),
        # 平移、缩放和旋转
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.05, rotate_limit=10, p=0.5),
        # 选择以下图像变换中的一种
        A.OneOf([
            # 网格扭曲
            A.GridDistortion(num_steps=5, distort_limit=0.05, p=1.0),
            # 弹性变换
            A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=1.0)
        ], p=0.25),
        # 随机去除图像的部分区域
        A.CoarseDropout(
            max_holes=8,
            max_height=CFG.img_size[0] // 20,
            max_width=CFG.img_size[1] // 20,
            min_holes=5,
            fill_value=0,
            mask_fill_value=0,
            p=0.5
        ),
    ], p=1.0),

    "valid": A.Compose([
        # 调整图像大小
        A.Resize(*CFG.img_size, interpolation=cv2.INTER_NEAREST),
    ], p=1.0)
}
