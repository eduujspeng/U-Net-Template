# -*- coding:utf-8 -*-
# author:peng
# Date：2024/1/15 9:37
# wandb是Weight & Bias的缩写，这是一个与Tensorboard类似的参数可视化平台。不过，相比较TensorBoard而言，Wandb更加的强大
# Wandb支持更多的数据类型，例如文本、图像、音频等。
# Wandb支持更加丰富的可视化功能，例如直方

import wandb

from utils.config import CFG


def wandb_init():
    try:
        # 不需要手动登录，W&B 会自动使用配置文件或环境变量中的密钥
        # 如果需要，可以通过配置文件（或环境变量）设置 WANDB_API_KEY, 不要翻墙
        # wandb.login(key='your-api-key')

        # 设置W&B初始化参数
        run = wandb.init(
            project='U-Net-template-DRIVE',
            config={k: v for k, v in dict(vars(CFG)).items() if '__' not in k},
            name=f"{CFG.img_size[0]}x{CFG.img_size[1]}|model-{CFG.model_name}",
            group=CFG.comment,
        )
        return run
    except wandb.Error as e:
        CFG.wandb_flag = 'Must'
        print(
            f'Error during W&B initialization: {e}\n'
            'To use your W&B account, please make sure your W&B API key is set.\n'
            'You can set your W&B API key by adding it to your secrets in the W&B project settings.\n'
            'Get your W&B API key here: https://wandb.ai/authorize'
        )
        return None


if __name__ == '__main__':
    run = wandb_init()
    if run:
        print(run)
    else:
        print('W&B initialization failed.')
