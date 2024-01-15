# -*- coding:utf-8 -*-
# author:peng
# Date：2024/1/15 10:22
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

# 定义 BuildDataset 类，继承自 torch.utils.data.Dataset
from utils.config import CFG
from utils.util import load_img, load_msk, data_transforms, plot_batch


class BuildDataset(Dataset):
    # 初始化方法，接收数据框 df、标签标志 label、和变换 transforms
    def __init__(self, df, label=True, transforms=None):
        self.df = df
        self.label = label
        self.img_paths = df['image_path'].tolist()
        self.msk_paths = df['mask_path'].tolist()
        self.transforms = transforms

    # 返回数据集的长度
    def __len__(self):
        return len(self.df)

    # 根据给定的索引返回对应的样本
    def __getitem__(self, index):
        # 获取图像路径
        img_path = self.img_paths[index]

        # 调用 load_img 函数加载图像
        img = load_img(img_path)

        # 如果 label 为 True，表示需要加载标签
        if self.label:
            # 获取掩码路径
            msk_path = self.msk_paths[index]

            # 调用 load_msk 函数加载掩码
            msk = load_msk(msk_path)

            # 如果提供了数据变换函数 transforms，则应用变换
            if self.transforms:
                data = self.transforms(image=img, mask=msk)
                img = data['image']
                msk = data['mask']

            # 将图像和掩码的通道维度调整为 (C, H, W) 的形状
            img = np.transpose(img, (2, 0, 1))
            msk = np.transpose(msk, (2, 0, 1))

            # 返回图像和掩码的 PyTorch 张量
            return torch.tensor(img), torch.tensor(msk)
        else:
            # 如果不需要加载标签，只加载图像
            if self.transforms:
                data = self.transforms(image=img)
                img = data['image']

            # 调整图像通道维度为 (C, H, W) 的形状
            img = np.transpose(img, (2, 0, 1))

            # 返回图像的 PyTorch 张量
            return torch.tensor(img)


# 定义函数 prepare_loaders，用于准备训练和验证数据加载器
def prepare_loaders(train_df, valid_df, debug=False):
    # 创建训练集和验证集的数据集实例，应用相应的数据变换
    train_dataset = BuildDataset(train_df, transforms=data_transforms['train'])
    valid_dataset = BuildDataset(valid_df, transforms=data_transforms['valid'])

    # 创建训练集和验证集的数据加载器
    train_loader = DataLoader(train_dataset, batch_size=CFG.train_bs if not debug else 20,
                              num_workers=4, shuffle=True, pin_memory=True, drop_last=False)
    valid_loader = DataLoader(valid_dataset, batch_size=CFG.valid_bs if not debug else 20,
                              num_workers=4, shuffle=False, pin_memory=True)

    # 返回训练集和验证集的数据加载器
    return train_loader, valid_loader


if __name__ == '__main__':
    df = pd.read_csv('../datasets/DRIVE/train.csv')
    train_loader, valid_loader = prepare_loaders(df, df)
    imgs, msks = next(iter(train_loader))
    print(imgs.size(), msks.size())
    plot_batch(imgs, msks, 5)
