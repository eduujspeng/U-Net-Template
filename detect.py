import pandas as pd
import torch
from torch import nn

from utils.build_dataset import prepare_loaders
from utils.config import CFG
from utils.util import load_model, plot_batch

if __name__ == '__main__':
    model = load_model('weight/best_epoch.bin')
    model.eval()

    df = pd.read_csv('datasets/DRIVE/train.csv')
    train_loader, valid_loader = prepare_loaders(df, df)

    imgs,msks = next(iter(valid_loader))
    imgs = imgs.to(CFG.device, dtype=torch.float)

    preds = []

    with torch.no_grad():
        pred = model(imgs)
        pred = (nn.Sigmoid()(pred) > 0.5).double()
    preds.append(pred)

    imgs = imgs.cpu().detach()
    preds = torch.mean(torch.stack(preds, dim=0), dim=0).cpu().detach()

    plot_batch(imgs, preds, size=5)
    plot_batch(imgs, msks, size=5)
