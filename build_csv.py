# -*- coding:utf-8 -*-
# author:peng
# Date：2024/1/15 10:31
# 图片路径存储至csv
import glob
import os

import pandas as pd

train_path_label = 'datasets/DRIVE/training/1st_manual'
train_path_image = 'datasets/DRIVE/training/images'

labels_path = sorted(os.listdir(train_path_label))
images_path = sorted(os.listdir(train_path_image))

print(labels_path)
print(images_path)

labels = sorted(glob.glob(train_path_label+'/*.gif'))
images = sorted(glob.glob(train_path_image+'/*.tif'))

print(labels)
print(images)

df = pd.DataFrame({'image_path': images, 'mask_path': labels})
print(df.head())


df.to_csv('datasets/DRIVE/train.csv', index=False)
