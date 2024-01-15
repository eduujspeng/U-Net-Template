# -*- coding:utf-8 -*-
# author:peng
# Date：2024/1/15 15:18
import copy
import gc
import time
from collections import defaultdict
import warnings
import numpy as np
import pandas as pd
import torch
import wandb
from torch import optim

from model.UNet import UNet
from utils.build_dataset import prepare_loaders
from utils.optimizer import fetch_scheduler
from utils.train_one_epoch import train_one_epoch, valid_one_epoch
from utils.config import CFG
from utils.wandb_login import wandb_init
warnings.filterwarnings('ignore')


def run_training(model, optimizer, scheduler, run, num_epochs, train_loader, valid_loader):
    # 通过 wandb.watch 自动记录梯度
    global best_jaccard
    wandb.watch(model, log_freq=100)

    # 如果有可用的 CUDA 设备，打印设备信息
    if torch.cuda.is_available():
        print("cuda: {}\n".format(torch.cuda.get_device_name()))

    # 记录训练过程的变量
    start = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_dice = -np.inf
    best_epoch = -1
    history = defaultdict(list)

    # 遍历每个 epoch
    for epoch in range(1, num_epochs + 1):
        gc.collect()
        print(f'Epoch {epoch}/{num_epochs}', end='')

        # 训练一个 epoch，并记录训练损失
        train_loss = train_one_epoch(model, optimizer, scheduler,
                                     dataloader=train_loader,
                                     device=CFG.device, epoch=epoch)

        # 验证一个 epoch，并记录验证损失和指标
        val_loss, val_scores = valid_one_epoch(model, optimizer, valid_loader,
                                               device=CFG.device,
                                               epoch=epoch)
        val_dice, val_jaccard = val_scores

        # 记录训练和验证指标
        history['Train Loss'].append(train_loss)
        history['Valid Loss'].append(val_loss)
        history['Valid Dice'].append(val_dice)
        history['Valid Jaccard'].append(val_jaccard)

        # 使用 wandb 记录训练和验证指标
        wandb.log({"Train Loss": train_loss,
                   "Valid Loss": val_loss,
                   "Valid Dice": val_dice,
                   "Valid Jaccard": val_jaccard,
                   "LR": scheduler.get_last_lr()[0]})

        # 打印验证 Dice 和 Jaccard 指标
        print(f'Valid Dice: {val_dice:0.4f} | Valid Jaccard: {val_jaccard:0.4f}')

        # 更新最佳模型权重
        if val_dice >= best_dice:
            print(f"Valid Score Improved ({best_dice:0.4f} ---> {val_dice:0.4f})")
            best_dice = val_dice
            best_jaccard = val_jaccard
            best_epoch = epoch
            run.summary["Best Dice"] = best_dice
            run.summary["Best Jaccard"] = best_jaccard
            run.summary["Best Epoch"] = best_epoch
            best_model_wts = copy.deepcopy(model.state_dict())
            PATH = f"./weight/best_epoch.bin"
            torch.save(model.state_dict(), PATH)
            # 保存模型文件到当前目录
            wandb.save(PATH)
            print(f"Model Saved")

        # 保存当前模型权重
        last_model_wts = copy.deepcopy(model.state_dict())
        PATH = f"last_epoch.bin"
        torch.save(model.state_dict(), PATH)

        print()

    # 训练完成，计算总体时间
    end = time.time()
    time_elapsed = end - start
    print('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(
        time_elapsed // 3600, (time_elapsed % 3600) // 60, (time_elapsed % 3600) % 60))
    print("Best Score: {:.4f}".format(best_jaccard))

    # 加载最佳模型权重
    model.load_state_dict(best_model_wts)

    # 返回训练完成的模型和训练历史记录
    return model, history


if __name__ == '__main__':
    model = UNet(in_channels=3, out_channels=1).to(CFG.device)
    optimizer = optim.Adam(model.parameters(), lr=CFG.lr, weight_decay=CFG.wd)
    scheduler = fetch_scheduler(optimizer)

    run = wandb_init()
    df = pd.read_csv('datasets/DRIVE/train.csv')
    train_loader, valid_loader = prepare_loaders(df, df)

    run_training(model, optimizer, scheduler, run, CFG.epochs, train_loader, valid_loader)

    # 结束 wandb 实验
    run.finish()

