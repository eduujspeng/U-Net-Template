# -*- coding:utf-8 -*-
# author:peng
# Date：2024/1/15 10:04
import torch


class CFG:
    wandb_flag = False
    seed = 42
    debug = False  # 设置 debug=False 进行完整训练
    exp_name = 'Baselinev2'  # 实验名称
    comment = 'unet-efficientnet_b1-224x224-aug2-split2'  # 实验备注
    model_name = 'UNet'  # 模型名称
    backbone = 'efficientnet-b1'  # 模型骨干网络
    train_bs = 128  # 训练时的批量大小
    valid_bs = train_bs * 2  # 验证时的批量大小，通常是训练批量大小的两倍
    img_size = [224, 224]  # 输入图像的大小
    epochs = 1  # 训练的总轮次
    lr = 2e-3  # 初始学习率
    scheduler = 'CosineAnnealingLR'  # 学习率调度器的类型
    min_lr = 1e-6  # 学习率的最小值
    T_max = int(30000 / train_bs * epochs) + 50  # CosineAnnealingLR 调度器的周期
    T_0 = 25  # CosineAnnealingWarmRestarts 调度器的初始周期
    warmup_epochs = 0  # 学习率预热的轮次
    wd = 1e-6  # 权重衰减
    n_accumulate = max(1, 32 // train_bs)  # 累积梯度的步数，用于增大批量大小
    n_fold = 5  # 交叉验证的折数
    num_classes = 1  # 类别数
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 训练设备
