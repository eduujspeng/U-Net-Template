# U-Net Training Template with wandb Logging

## Overview

This repository contains a U-Net training template integrated with Weights & Biases (wandb) for efficient tracking and visualization of training metrics. U-Net is widely used for image segmentation tasks, and wandb is an excellent tool for experiment tracking.

## Prerequisites

- Python 3.x
- PyTorch
- wandb account

## Installation

1. Clone this repository.
2. Install the required packages:
   ``` python
   pip install torch torchvision wandb
   ```
4. Login to wandb:
``` python
wandb login
```

## Data Preparation

Organize your dataset into `data/` directory with `train/`, `val/`, and `test/` subdirectories.

## Configuration

Edit `config.py` to set up your training parameters and wandb configurations.

## Training

Execute the training script:
```
python train.py
```


## wandb Integration

Key integration points in `train.py`:

- Initialize wandb:
  ```python
  wandb.init(project="unet_segmentation")
  ```

Log training metrics:
```
wandb.log({"loss": loss_value, "accuracy": accuracy_value})
```

Log prediction images:
```
wandb.log({"predictions": wandb.Image(prediction_image, caption="Prediction")})
```

## Results
Monitor training metrics and visualize results in real-time on your wandb dashboard.

## Customization
Feel free to customize the U-Net model, training loop, and wandb settings as per your project requirements.

## Issues and Support
For any issues or questions, please open an issue in this repository.



