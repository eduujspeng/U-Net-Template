# -*- coding:utf-8 -*-
# author:peng
# Date：2024/1/15 19:14
'''
这个函数接受一个优化器 optimizer 作为参数，并根据配置文件中指定的 CFG.scheduler 的值选择相应的学习率调度器。
支持的调度器有 CosineAnnealingLR、CosineAnnealingWarmRestarts、ReduceLROnPlateau、ExponentialLR。
如果配置文件中的 CFG.scheduler 为 None，表示不使用学习率调度器，则返回 None。函数最终返回所选的学习率调度器。
'''
from torch.optim import lr_scheduler

from utils.config import CFG


def fetch_scheduler(optimizer):
    # 根据配置选择不同的学习率调度器
    global scheduler
    if CFG.scheduler == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG.T_max,
                                                   eta_min=CFG.min_lr)
    elif CFG.scheduler == 'CosineAnnealingWarmRestarts':
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=CFG.T_0,
                                                             eta_min=CFG.min_lr)
    elif CFG.scheduler == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,
                                                   mode='min',
                                                   factor=0.1,
                                                   patience=7,
                                                   threshold=0.0001,
                                                   min_lr=CFG.min_lr, )
    elif CFG.scheduler == 'ExponentialLR':
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.85)
    elif CFG.scheduler is None:
        return None

    return scheduler
