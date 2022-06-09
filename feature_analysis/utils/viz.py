import random
from typing import List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torchvision


def plot_sample_predictions(model: pl.LightningModule, datamodule: pl.LightningDataModule, samples: int = 4):
    model.eval()
    with torch.no_grad():
        for data in random.sample(list(datamodule.predict_dataloader()), 1):
            imgs, labels = data
            count = samples
            if samples > 1:
                s_imgs = torchvision.utils.make_grid(imgs[:count])
            else:
                count = 1
                s_imgs = imgs[0]
            s_imgs = np.transpose(s_imgs.cpu().numpy(), [1, 2, 0])

            plt.imshow(s_imgs)
            predictions = model.model(imgs).cpu().numpy()
            pred_y = np.argmax(predictions, axis=-1)
            plt.title(f"label={labels.cpu().numpy()[:count]}\npredicted={pred_y[:count]}")


def plot_torch_image(
    imgs: Union[List, np.ndarray], labels: np.ndarray, sample: int = 1, model: pl.LightningModule = None
):
    if isinstance(imgs, List):
        imgs = np.vstack(imgs)

    if not isinstance(imgs, torch.Tensor):
        imgs, labels = torch.from_numpy(imgs), torch.from_numpy(labels)

    s_imgs = torchvision.utils.make_grid(imgs[:sample])
    s_imgs = np.transpose(s_imgs.cpu().numpy(), [1, 2, 0])
    plt.imshow(s_imgs)
    title = f"label={labels.cpu().numpy()[:sample]}"
    if model:
        predictions = model.model(imgs).cpu().numpy()
        pred_y = np.argmax(predictions, axis=-1)
        title = f"{title}\npredicted={pred_y[:sample]}"
    plt.title(title)


def plot_raw_conv_features(conv_fm: np.array, new_shape: Tuple):
    grid_img = torchvision.utils.make_grid(torch.from_numpy(conv_fm[:, None, ...]), nrow=4)
    plt.imshow(grid_img.permute(1, 2, 0))
