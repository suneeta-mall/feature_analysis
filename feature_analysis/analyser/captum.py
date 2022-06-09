import random
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torchvision
from captum.attr import DeepLift, GradientShap, IntegratedGradients, NoiseTunnel, Occlusion, Saliency
from captum.attr import visualization as viz

__all__ = [
    "CaptumExplainer",
]


class CaptumExplainer:
    def __init__(self, model: pl.LightningModule, datamodule: pl.LightningDataModule) -> None:
        self.model = model
        self.datamodule = datamodule

    def extract_random(self):

        model = self.model.model
        model.eval()
        for data in random.sample(list(self.datamodule.predict_dataloader()), 1):
            imgs, labels = data
            image = imgs[0]
            label = labels[0]
            image = image[None, :]
            image.requires_grad = True

            s_imgs = torchvision.utils.make_grid(imgs[:1])
            s_imgs = np.transpose(s_imgs, [1, 2, 0])
            predictions = model(imgs)
            pred_y = torch.argmax(predictions, axis=-1)

            saliency = Saliency(model)
            grads = saliency.attribute(image, target=label.item())
            grads = np.transpose(torchvision.utils.make_grid(grads[:1]), [1, 2, 0])

            ig = IntegratedGradients(model)
            attr_ig, delta = ig.attribute(image, target=label, baselines=image * 0, return_convergence_delta=True)
            attr_ig = np.transpose(torchvision.utils.make_grid(attr_ig[:1]), [1, 2, 0])

            nt = NoiseTunnel(ig)
            (attr_nt_ig,) = nt.attribute(
                image, target=label, baselines=image * 0, nt_type="smoothgrad_sq", nt_samples=100, stdevs=0.2
            )
            attr_nt_ig = np.transpose(torchvision.utils.make_grid(attr_nt_ig[:1]), [1, 2, 0])

            dl = DeepLift(model)
            attr_dl = dl.attribute(image, target=label, baselines=image * 0)
            attr_dl = np.transpose(torchvision.utils.make_grid(attr_dl[:1]), [1, 2, 0])

            gradient_shap = GradientShap(model)
            rand_img_dist = torch.cat([image * 0, image * 1])
            attr_gs = gradient_shap.attribute(
                image,
                target=label,
                n_samples=50,
                stdevs=0.0001,
                baselines=rand_img_dist,
            )
            attr_gs = np.transpose(torchvision.utils.make_grid(attr_gs[:1]), [1, 2, 0])

            occlusion = Occlusion(model)
            attr_occ = occlusion.attribute(
                image, target=label, strides=(1, 3, 3), sliding_window_shapes=(1, 5, 5), baselines=0
            )
            attr_occ = np.transpose(torchvision.utils.make_grid(attr_occ[:1]), [1, 2, 0])

        return {
            "s_imgs": s_imgs,
            "labels": labels[:1],
            "pred_y": pred_y[:1],
            "grads": grads,
            "attr_ig": attr_ig,
            "delta": delta,
            "attr_nt_ig": attr_nt_ig,
            "attr_dl": attr_dl,
            "attr_gs": attr_gs,
            "attr_occ": attr_occ,
        }

    def extract(self, imgs, labels):
        if isinstance(imgs, List):
            imgs = np.vstack(imgs)

        if not isinstance(imgs, torch.Tensor):
            imgs, labels = torch.from_numpy(imgs), torch.from_numpy(labels)

        model = self.model.model
        model.eval()
        image = imgs[0]
        label = labels[0]
        image = image[None, :]
        image.requires_grad = True

        s_imgs = torchvision.utils.make_grid(imgs[:1])
        s_imgs = np.transpose(s_imgs, [1, 2, 0])
        predictions = model(imgs)
        pred_y = torch.argmax(predictions, axis=-1)

        saliency = Saliency(model)
        grads = saliency.attribute(image, target=label.item())
        grads = np.transpose(torchvision.utils.make_grid(grads[:1]), [1, 2, 0])

        ig = IntegratedGradients(model)
        attr_ig, delta = ig.attribute(image, target=label, baselines=image * 0, return_convergence_delta=True)
        attr_ig = np.transpose(torchvision.utils.make_grid(attr_ig[:1]), [1, 2, 0])

        nt = NoiseTunnel(ig)
        (attr_nt_ig,) = nt.attribute(
            image, target=label, baselines=image * 0, nt_type="smoothgrad_sq", nt_samples=100, stdevs=0.2
        )
        attr_nt_ig = np.transpose(torchvision.utils.make_grid(attr_nt_ig[:1]), [1, 2, 0])

        dl = DeepLift(model)
        attr_dl = dl.attribute(image, target=label, baselines=image * 0)
        attr_dl = np.transpose(torchvision.utils.make_grid(attr_dl[:1]), [1, 2, 0])

        gradient_shap = GradientShap(model)
        rand_img_dist = torch.cat([image * 0, image * 1])
        attr_gs = gradient_shap.attribute(
            image,
            target=label,
            n_samples=50,
            stdevs=0.0001,
            baselines=rand_img_dist,
        )
        attr_gs = np.transpose(torchvision.utils.make_grid(attr_gs[:1]), [1, 2, 0])

        occlusion = Occlusion(model)
        attr_occ = occlusion.attribute(
            image, target=label, strides=(1, 3, 3), sliding_window_shapes=(1, 5, 5), baselines=0
        )
        attr_occ = np.transpose(torchvision.utils.make_grid(attr_occ[:1]), [1, 2, 0])

        return {
            "s_imgs": s_imgs,
            "labels": labels[:1],
            "pred_y": pred_y[:1],
            "grads": grads,
            "attr_ig": attr_ig,
            "delta": delta,
            "attr_nt_ig": attr_nt_ig,
            "attr_dl": attr_dl,
            "attr_gs": attr_gs,
            "attr_occ": attr_occ,
        }

    def plt_plot(self, result_dict: Dict):
        f = plt.figure(figsize=(20, 20))
        plt.subplot(171)
        plt.imshow(result_dict["s_imgs"])
        plt.title(f"label={result_dict['labels']}\npredicted={result_dict['pred_y']}")

        plt.subplot(172)
        plt.imshow(result_dict["grads"])

        plt.subplot(173)
        plt.imshow(result_dict["attr_ig"])
        plt.title(f"IG:~delta={abs(result_dict['delta'])}")

        plt.subplot(174)
        plt.imshow(result_dict["attr_nt_ig"])
        plt.title("IG: Noisesmooth")

        plt.subplot(175)
        plt.imshow(result_dict["attr_dl"])
        plt.title("DeepLift")

        plt.subplot(176)
        plt.imshow(result_dict["attr_gs"])
        plt.title("GradientShap")

        plt.subplot(177)
        plt.imshow(result_dict["attr_occ"])
        plt.title("Occlusion")

    def visualize(self, result_dict: Dict):
        original_image = result_dict["s_imgs"].detach().numpy()

        viz.visualize_image_attr(None, original_image, method="original_image", title="Image")

        viz.visualize_image_attr(
            result_dict["grads"].detach().numpy(),
            original_image,
            method="blended_heat_map",
            sign="absolute_value",
            show_colorbar=True,
            title="Gradient Magnitudes",
        )

        viz.visualize_image_attr(
            result_dict["attr_ig"].detach().numpy(),
            original_image,
            method="blended_heat_map",
            sign="all",
            show_colorbar=True,
            title="Integrated Gradients",
        )

        viz.visualize_image_attr(
            result_dict["attr_nt_ig"].detach().numpy(),
            original_image,
            method="blended_heat_map",
            sign="absolute_value",
            outlier_perc=10,
            show_colorbar=True,
            title="Overlayed Integrated Gradients with SmoothGrad Squared",
        )

        viz.visualize_image_attr(
            result_dict["attr_dl"].detach().numpy(),
            original_image,
            method="blended_heat_map",
            sign="all",
            show_colorbar=True,
            title="Overlayed DeepLift",
        )

        viz.visualize_image_attr(
            result_dict["attr_gs"].detach().numpy(),
            original_image,
            method="blended_heat_map",
            sign="all",
            show_colorbar=True,
            title="Overlayed GradientShap",
        )

        viz.visualize_image_attr(
            result_dict["attr_occ"].detach().numpy(),
            original_image,
            method="blended_heat_map",
            sign="all",
            show_colorbar=True,
            title="Overlayed Occlusion",
        )
