from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import offsetbox

# pylint: disable=unused-wildcard-import,wildcard-import
from plotnine import *

# pylint: enable=unused-wildcard-import,wildcard-import
from sklearn.manifold import TSNE

__all__ = [
    "TSNEExplorer",
]


class TSNEExplorer:
    def __init__(self) -> None:
        pass

    def logits_layer_weights(
        self,
        X: np.ndarray,
        classes_list: List[int],
        perplexity: int = 7,
        learning_rate: float = 100.0,
        method: str = "barnes_hut",
        init: str = None,
        n_iter: int = 1050,
    ) -> pd.DataFrame:
        tsne = TSNE(
            n_components=2, perplexity=perplexity, learning_rate=learning_rate, method=method, init=init, n_iter=n_iter
        )
        tsne = tsne.fit_transform(X)

        df = pd.DataFrame(data=tsne, columns=["TSNE-C1", "TSNE-C2"])
        df["class"] = classes_list

        return df

    def feature_map(
        self,
        X: np.ndarray,
        num_output: List[int],
        perplexity: int = 7,
        learning_rate: float = 100.0,
        method: str = "barnes_hut",
        init: str = "pca",
        n_iter: int = 1050,
    ) -> pd.DataFrame:
        tsne = TSNE(
            n_components=2, perplexity=perplexity, learning_rate=learning_rate, method=method, init=init, n_iter=n_iter
        )
        tsne = tsne.fit_transform(X)

        df = pd.DataFrame(data=tsne, columns=["TSNE-C1", "TSNE-C2"])
        df["class"] = num_output

        return df

    def feature_map_batch(
        self,
        X: np.ndarray,
        r_labels_set: np.ndarray,
        perplexity: int = 7,
        learning_rate: float = 100.0,
        method: str = "barnes_hut",
        init: str = "pca",
        n_iter: int = 1050,
    ) -> pd.DataFrame:
        tsne = TSNE(
            n_components=2, perplexity=perplexity, learning_rate=learning_rate, method=method, init=init, n_iter=n_iter
        )
        tsne = tsne.fit_transform(X)

        df = pd.DataFrame(data=tsne, columns=["TSNE-C1", "TSNE-C2"])
        df["class"] = r_labels_set
        return df

    def plot_gg(self, df: pd.DataFrame) -> None:
        return (
            ggplot(df)
            + geom_point(aes(x="TSNE-C1", y="TSNE-C2", colour="class"), size=8)
            + geom_text(aes(x="TSNE-C1", y="TSNE-C2", label="class"), size=10)
            + theme(figure_size=(16, 16))
        )

    def plot_with_images(self, df, labels=None, images=None, ax=None, thumb_frac=0.05, cmap="gray"):
        proj = df.to_numpy()
        plt.figure(figsize=(30, 30))

        ax = ax or plt.gca()
        ax.scatter(proj[:, 0], proj[:, 1], c=labels, s=50)

        if images is not None:
            min_dist_2 = (thumb_frac * max(proj.max(0) - proj.min(0))) ** 2
            shown_images = np.array([2 * proj.max(0)])
            for i in range(proj.shape[0]):
                dist = np.sum((proj[i] - shown_images) ** 2, 1)
                if np.min(dist) < min_dist_2:
                    # don't show points that are too close
                    continue
                shown_images = np.vstack([shown_images, proj[i]])
                img = np.transpose(images[i][0], [1, 2, 0])
                imagebox = offsetbox.AnnotationBbox(offsetbox.OffsetImage(img, cmap=cmap), proj[i][:2])
                ax.add_artist(imagebox)
        ax.legend()
