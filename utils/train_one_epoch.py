# -*- coding:utf-8 -*-
# author:peng
# Date：2024/1/15 9:33
import gc

import numpy as np
import torch
from torch import nn
from tqdm import tqdm
import torch.cuda.amp as amp

from utils.config import CFG
from utils.criterion import criterion, dice_coef, iou_coef


# 定义训练一个 epoch 的函数
def train_one_epoch(model, optimizer, scheduler, dataloader, device, epoch):
    # 设置模型为训练模式
    global epoch_loss
    cuda_flag = True if CFG.device=='cuda:0' else False
    model.train()

    # 只有在CUDA可用时才使用混合精度训练（amp），否则进行普通的训练。
    scaler = amp.GradScaler(enabled=cuda_flag)

    dataset_size = 0
    running_loss = 0.0

    # 使用 tqdm 显示训练进度
    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc='Train ')

    # 遍历数据加载器
    for step, (images, masks) in pbar:
        images = images.to(device, dtype=torch.float)
        masks = masks.to(device, dtype=torch.float)

        batch_size = images.size(0)

        # 使用混合精度进行前向传播和计算损失
        with amp.autocast(enabled=cuda_flag):
            y_pred = model(images)
            loss = criterion(y_pred, masks)
            loss = loss / CFG.n_accumulate

        if cuda_flag:
            # 使用混合精度进行反向传播
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (step + 1) % CFG.n_accumulate == 0:
            if cuda_flag:
                # 使用混合精度进行参数更新
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            # zero the parameter gradients
            optimizer.zero_grad()

            # 如果使用学习率调度器，则进行调度
            if scheduler is not None:
                scheduler.step()

        # 计算累计损失和样本数
        running_loss += (loss.item() * batch_size)
        dataset_size += batch_size

        # 计算平均 epoch 损失
        epoch_loss = running_loss / dataset_size

        # 在 tqdm 中更新训练进度显示
        mem = torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0
        current_lr = optimizer.param_groups[0]['lr']
        pbar.set_postfix(train_loss=f'{epoch_loss:0.4f}',
                         lr=f'{current_lr:0.5f}',
                         gpu_mem=f'{mem:0.2f} GB')

    # 释放 GPU 内存
    if cuda_flag:
        torch.cuda.empty_cache()
    gc.collect()

    # 返回平均 epoch 损失
    return epoch_loss


@torch.no_grad()
def valid_one_epoch(model, optimizer, dataloader, device, epoch):
    # 设置模型为评估模式
    global epoch_loss
    model.eval()

    dataset_size = 0
    running_loss = 0.0

    val_scores = []

    # 使用 tqdm 显示验证进度
    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc='Valid ')

    # 遍历数据加载器
    for step, (images, masks) in pbar:
        images = images.to(device, dtype=torch.float)
        masks = masks.to(device, dtype=torch.float)

        batch_size = images.size(0)

        # 进行前向传播和计算损失
        y_pred = model(images)
        loss = criterion(y_pred, masks)

        # 计算累计损失和样本数
        running_loss += (loss.item() * batch_size)
        dataset_size += batch_size

        # 计算平均 epoch 损失
        epoch_loss = running_loss / dataset_size

        # 对预测进行 Sigmoid 操作，然后计算 Dice 系数和 Jaccard 系数
        y_pred = nn.Sigmoid()(y_pred)
        val_dice = dice_coef(masks, y_pred).cpu().detach().numpy()
        val_jaccard = iou_coef(masks, y_pred).cpu().detach().numpy()
        val_scores.append([val_dice, val_jaccard])

        # 在 tqdm 中更新验证进度显示
        mem = torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0
        current_lr = optimizer.param_groups[0]['lr']
        pbar.set_postfix(valid_loss=f'{epoch_loss:0.4f}',
                         lr=f'{current_lr:0.5f}',
                         gpu_memory=f'{mem:0.2f} GB')

    # 计算平均验证集 Dice 系数和 Jaccard 系数
    val_scores = np.mean(val_scores, axis=0)

    # 释放 GPU 内存
    torch.cuda.empty_cache()
    gc.collect()

    # 返回平均 epoch 损失和验证指标
    return epoch_loss, val_scores
