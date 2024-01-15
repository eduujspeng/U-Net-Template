# -*- coding:utf-8 -*-
# author:peng
# Date：2024/1/15 9:35
import torch
import torch.nn as nn


# 定义U-Net网络的编码器部分
class UNetEncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetEncoderBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu2(x)
        skip = x  # 保留编码器阶段的特征图，用于解码器阶段的跳跃连接
        x = self.pool(x)
        return x, skip


# 定义U-Net网络的解码器部分
class UNetDecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetDecoderBlock, self).__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x, skip):
        x = self.upconv(x)
        x = torch.cat([x, skip], dim=1)  # 使用跳跃连接连接解码器的输入和相应的编码器特征图
        x = self.conv(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu2(x)
        return x


# 定义完整的U-Net模型
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()

        # 编码器阶段
        self.enc1 = UNetEncoderBlock(in_channels, 64)
        self.enc2 = UNetEncoderBlock(64, 128)
        self.enc3 = UNetEncoderBlock(128, 256)
        self.enc4 = UNetEncoderBlock(256, 512)

        # 中间部分
        self.middle = nn.Conv2d(512, 1024, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

        # 解码器阶段
        self.dec4 = UNetDecoderBlock(1024, 512)
        self.dec3 = UNetDecoderBlock(512, 256)
        self.dec2 = UNetDecoderBlock(256, 128)
        self.dec1 = UNetDecoderBlock(128, 64)

        # 输出层
        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # 编码器阶段
        enc1, skip1 = self.enc1(x)
        enc2, skip2 = self.enc2(enc1)
        enc3, skip3 = self.enc3(enc2)
        enc4, skip4 = self.enc4(enc3)

        # 中间部分
        middle = self.middle(enc4)
        middle = self.relu(middle)

        # 解码器阶段
        dec4 = self.dec4(middle, skip4)
        dec3 = self.dec3(dec4, skip3)
        dec2 = self.dec2(dec3, skip2)
        dec1 = self.dec1(dec2, skip1)

        # 输出层
        output = self.out_conv(dec1)

        return output


if __name__ == '__main__':
    # 创建U-Net模型实例
    model = UNet(in_channels=3, out_channels=1)
    print(model)
    inputs = torch.randn(1, 3, 256, 256)
    outputs = model(inputs)
    print(outputs.shape)
