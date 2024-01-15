# -*- coding:utf-8 -*-
# author:peng
# Date：2024/1/15 15:10
import torch


# 定义计算 Dice 系数的函数
def dice_coef(y_true, y_pred, thr=0.5, dim=(2, 3), epsilon=0.001):
    y_true = y_true.to(torch.float32)
    y_pred = (y_pred > thr).to(torch.float32)
    inter = (y_true * y_pred).sum(dim=dim)
    den = y_true.sum(dim=dim) + y_pred.sum(dim=dim)
    dice = ((2 * inter + epsilon) / (den + epsilon)).mean(dim=(1, 0))
    return dice


# 定义计算 IoU 系数的函数
def iou_coef(y_true, y_pred, thr=0.5, dim=(2, 3), epsilon=0.001):
    y_true = y_true.to(torch.float32)
    y_pred = (y_pred > thr).to(torch.float32)
    inter = (y_true * y_pred).sum(dim=dim)
    union = (y_true + y_pred - y_true * y_pred).sum(dim=dim)
    iou = ((inter + epsilon) / (union + epsilon)).mean(dim=(1, 0))
    return iou


def criterion(inputs, target):
    smooth = 1e-6
    intersection = 2.0 * ((target * inputs).sum()) + smooth
    union = target.sum() + inputs.sum() + smooth

    inputs = torch.where(inputs >= 0.5, 1, 0)
    intersection1 = 2.0 * ((target * inputs).sum()) + smooth
    union1 = target.sum() + inputs.sum() + smooth

    return 1 - (intersection / union) + 1 - (intersection1 / union1)
