import base64
from io import BytesIO
from typing import List, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
import umap
import umap.plot
from bokeh.models import HoverTool
from PIL import Image
from umap.parametric_umap import ParametricUMAP

__all__ = [
    "UMapExplorer",
]


class UMapExplorer:
    def __init__(self) -> None:
        self.n_components = 2
        self.hover_tool = HoverTool(
            tooltips="""
<div>
    <div>
        <img src='@image' style='float: left; margin: 5px 5px 5px 5px'/>
    </div>
    <div>
        <span style='font-size: 16px; color: #224499'>Class:</span>
        <span style='font-size: 18px'>@class</span>
    </div>
    <div>
        <span style='font-size: 16px; color: #224499'>Index:</span>
        <span style='font-size: 18px'>$index</span>
    </div>
    <div>
        <span style='font-size: 16px; color: #224499'>Values:</span>
        <span style='font-size: 18px'>($x, $y)</span>
    </div>
</div>
"""
        )

    def logits_layer_weights(
        self,
        X: np.ndarray,
        classes_list: List[int],
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        n_epochs: int = None,
        learning_rate: float = 1.0,
    ) -> pd.DataFrame:
        mapper = umap.UMAP(
            n_neighbors=n_neighbors, min_dist=min_dist, n_epochs=n_epochs, learning_rate=learning_rate
        ).fit(X)
        r_labels_set = np.array(classes_list)

        return mapper, r_labels_set

    def feature_map(
        self,
        X: np.ndarray,
        r_labels_set: np.ndarray,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        n_epochs: int = None,
        learning_rate: float = 1.0,
    ) -> pd.DataFrame:
        mapper = umap.UMAP(
            n_neighbors=n_neighbors, min_dist=min_dist, n_epochs=n_epochs, learning_rate=learning_rate
        ).fit(X)
        return mapper, np.array(r_labels_set)

    def feature_map_parametric(
        self, feature_map_set: np.ndarray, r_labels_set: np.ndarray, new_shape: Tuple, total_item: int
    ) -> pd.DataFrame:
        conv_feature_vector = feature_map_set.transpose(3, 0, 1, 2).reshape(total_item, -1)

        encoder = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=new_shape),
                tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=(2, 2), activation="relu", padding="same"),
                tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=(2, 2), activation="relu", padding="same"),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(units=256, activation="relu"),
                tf.keras.layers.Dense(units=256, activation="relu"),
                tf.keras.layers.Dense(units=self.n_components),
            ]
        )
        encoder.summary()

        embedder = ParametricUMAP(encoder=encoder, dims=new_shape)
        embedding = embedder.fit_transform(conv_feature_vector)
        return embedding

    def plot_gg(self, mapper: List, r_labels_set: List) -> None:
        return umap.plot.points(mapper, labels=r_labels_set, color_key_cmap="Paired", background="black")

    def plot_interactive(self, mapper: List, r_labels_set: List, image_data_set: List, notebook: bool = True):
        # Install this version of umap https://github.com/lmcinnes/umap/pull/858
        img_set = np.vstack(image_data_set)

        df = pd.DataFrame()
        df["image"] = list(map(self.embeddable_image, np.vsplit(img_set, img_set.shape[0])))
        df["class"] = r_labels_set.tolist()

        if notebook:
            umap.plot.output_notebook()

        p = umap.plot.interactive(
            mapper,
            labels=r_labels_set,
            hover_data=df,
            point_size=2,
            tools=["pan", "wheel_zoom", "box_zoom", "save", "reset", "help", self.hover_tool],
            color_key_cmap="Paired",
            background="black",
        )
        return umap.plot.show(p)

    @staticmethod
    def embeddable_image(data):
        img_data = (data[0, 0, ...] * 255).astype(np.uint8)
        image = Image.fromarray(img_data, mode="L").resize((64, 64), Image.BICUBIC)
        buffer = BytesIO()
        image.save(buffer, format="jpeg")
        for_encoding = buffer.getvalue()
        image_blurb = f"data:image/jpg;base64,{base64.b64encode(for_encoding).decode()}"
        return image_blurb
