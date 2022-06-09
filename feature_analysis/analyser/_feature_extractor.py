import itertools
import random
from typing import List, Tuple

import numpy as np
import pytorch_lightning as pl
import torch

__all__ = [
    "Extractor",
]


class Extractor:
    def __init__(self, model: pl.LightningModule, datamodule: pl.LightningDataModule) -> None:
        self.model = model
        self.datamodule = datamodule
        self._max_batch_sample = min(len(self.datamodule.predict_dataloader()), 20)

    def raw_data_batch(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray], Tuple]:

        fc3_features_set = None
        labels_set = None
        _new = True
        image_set = []

        for data in random.sample(list(self.datamodule.predict_dataloader()), self._max_batch_sample):
            imgs, labels = data
            fc3_features = imgs.cpu().numpy()
            if _new:
                fc3_features_set = fc3_features[np.newaxis, :]
                labels_set = labels.cpu().numpy()[np.newaxis, :]
                _new = False
            else:
                if fc3_features.shape[0] != fc3_features_set.shape[1]:
                    continue
                fc3_features_set = np.append(fc3_features_set, fc3_features[np.newaxis, :], axis=0)
                labels_set = np.append(labels_set, labels.cpu().numpy()[np.newaxis, :], axis=0)

            image_set.extend(np.vsplit(imgs, imgs.shape[0]))

        num_channel = len(fc3_features_set.shape)
        l = [[2], [0], [1]]
        if num_channel > 3:
            l = [[2], list(range(3, num_channel)), [0], [1]]
        l = list(itertools.chain(*l))
        new_shape = [fc3_features.shape[-1]] if len(fc3_features.shape) <= 2 else fc3_features.shape[-3:]
        r_fc3_features_set = fc3_features_set.transpose(*l).reshape(*new_shape, -1)
        r_labels_set = labels_set.reshape(-1)

        total_item = fc3_features_set.shape[0] * fc3_features_set.shape[1]

        return r_fc3_features_set, r_labels_set, image_set, (new_shape, total_item)

    def logits_layer_weights(self) -> Tuple[np.ndarray, List[int]]:

        num_classes = self.model.model.fc3.weight.shape[0]
        final_wts = self.model.model.fc3.weight.detach().numpy()

        X = np.asarray([np.abs(final_wts[i, :]) / sum(np.abs(final_wts[i, :])) for i in range(num_classes)])

        return X, list(range(num_classes))

    def feature_map(self, feature_layer_name: str = "fc3") -> Tuple[np.ndarray, List[int], Tuple]:
        activation = {}

        def get_activation(name):
            def hook(model, _input, output):
                activation[name] = output.detach()

            return hook

        model = self.model.model
        hook_handle = getattr(model, feature_layer_name).register_forward_hook(get_activation(feature_layer_name))

        model.eval()
        with torch.no_grad():
            for data in random.sample(list(self.datamodule.predict_dataloader()), 1):
                imgs, labels = data
                # img = np.transpose(imgs[0].cpu().numpy(), [1, 2, 0])
                predictions = model(imgs).cpu().numpy()
        features = activation[feature_layer_name].cpu().numpy()
        hook_handle.remove()

        num_output = features.shape[-1]
        X = np.asarray([np.abs(features[:, i]) / sum(np.abs(features[:, i])) for i in range(features.shape[-1])])
        return X, list(range(num_output)), data

    def feature_map_batch(
        self, feature_layer_name: str = "fc3"
    ) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray], Tuple]:
        activation = {}

        def get_activation(name):
            def hook(model, _input, output):
                activation[name] = output.detach()

            return hook

        model = self.model.model
        f3_hook_handle = getattr(model, feature_layer_name).register_forward_hook(get_activation(feature_layer_name))

        fc3_features_set = None
        labels_set = None
        _new = True
        image_set = []
        model.eval()
        with torch.no_grad():
            for data in random.sample(list(self.datamodule.predict_dataloader()), self._max_batch_sample):
                imgs, labels = data
                # img = np.transpose(imgs[0].cpu().numpy(), [1, 2, 0])
                predictions = model(imgs).cpu().numpy()
                fc3_features = activation[feature_layer_name].cpu().numpy()
                if _new:
                    fc3_features_set = fc3_features[np.newaxis, :]
                    labels_set = labels.cpu().numpy()[np.newaxis, :]
                    _new = False
                else:
                    if fc3_features.shape[0] != fc3_features_set.shape[1]:
                        continue
                    fc3_features_set = np.append(fc3_features_set, fc3_features[np.newaxis, :], axis=0)
                    labels_set = np.append(labels_set, labels.cpu().numpy()[np.newaxis, :], axis=0)

                image_set.extend(np.vsplit(imgs, imgs.shape[0]))

        f3_hook_handle.remove()
        num_channel = len(fc3_features_set.shape)
        l = [[2], [0], [1]]
        if num_channel > 3:
            l = [[2], list(range(3, num_channel)), [0], [1]]
        l = list(itertools.chain(*l))
        new_shape = [fc3_features.shape[-1]] if len(fc3_features.shape) <= 2 else fc3_features.shape[-3:]
        r_fc3_features_set = fc3_features_set.transpose(*l).reshape(*new_shape, -1)
        r_labels_set = labels_set.reshape(-1)

        total_item = fc3_features_set.shape[0] * fc3_features_set.shape[1]

        return r_fc3_features_set, r_labels_set, image_set, (new_shape, total_item)

    @staticmethod
    def normalize_features(r_fc3_features_set: np.ndarray) -> np.ndarray:
        X = np.asarray(
            [
                np.abs(r_fc3_features_set[..., i].ravel()) / sum(np.abs(r_fc3_features_set[..., i].ravel()))
                for i in range(r_fc3_features_set.shape[-1])
            ]
        )
        return X
