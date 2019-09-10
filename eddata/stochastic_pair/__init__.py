from edflow.util import PRNGMixin
from edflow.iterators.batches import DatasetMixin
from edflow.iterators.batches import resize_float32 as resize
from edflow.iterators.batches import load_image
import os
import numpy as np
from skimage.segmentation import slic
import pandas as pd
import eddata.utils as edu


class StochasticPairs(DatasetMixin, PRNGMixin):
    extracted_data_csv_columns = ["character_id", "relative_file_path_"]

    def __init__(self, config):
        """config has to include
        config: {
            "data_root": "foo",
            "data_csv": "train.csv",
            "spatial_size" : 256,
        }

        optional config parameters:
        config: {
            "data_flip" : False,        # flip data randomly. Default `False`.
            "avoid_identity" : False,   # avoid the identity. Default `False`.
            "data_csv_columns" : ["character_id", "relative_file_path_"] # `list` of `str` column names or "from_csv",
            "data_csv_has_header" : False # default `False`
        }

        suggested data_csv layout
        id,relative_image_path_,col3,co4,...
        for example:
        1,frames/01/00001.jpg,xxx,yyy
        1,frames/01/00002.jpg,xxx,yyy
        ...
        2,frames/02/00001.jpg,xxx,yyy
        2,frames/02/00002.jpg,xxx,yyy

        If the csv has more columns, the other columns will be ignored.
        Parameters
        ----------
        config: dict with options. See above

        Examples
        --------
            See test
        """
        self.config = config
        self.size = config["spatial_size"]
        self.root = config["data_root"]
        self.csv = config["data_csv"]
        self.csv_has_header = config.get("data_csv_has_header", False)
        self.avoid_identity = config.get("data_avoid_identity", True)
        self.flip = config.get("data_flip", False)
        self.make_labels()
        self.labels = edu.add_choices(self.labels)

    def make_labels(self):
        data_csv_columns = self.config.get(
            "data_csv_columns", self.extracted_data_csv_columns
        )
        if data_csv_columns == "from_csv":
            labels_df = pd.read_csv(self.csv)
            self.data_csv_columns = labels_df.columns
        else:
            self.data_csv_columns = data_csv_columns
            if self.csv_has_header:
                labels_df = pd.read_csv(self.csv)
            else:
                labels_df = pd.read_csv(self.csv, header=None)
            labels_df.rename(
                columns={
                    old: new
                    for old, new in zip(
                        labels_df.columns[: len(data_csv_columns)], data_csv_columns
                    )
                },
                inplace=True,
            )
        self.labels = dict(labels_df)
        self.labels = {k: list(v) for k, v in self.labels.items()}

        def add_root_path(x):
            return os.path.join(self.root, x)

        for label_name, i in zip(
            self.data_csv_columns, range(len(self.data_csv_columns))
        ):
            if "relative_" in label_name:
                label_update = {
                    label_name.replace("relative_", ""): list(
                        map(add_root_path, self.labels[label_name])
                    )
                }
                self.labels.update(label_update)
        self._length = len(self.labels)

    def __len__(self):
        return self._length

    def preprocess_image(self, image_path):
        image = load_image(image_path)
        image = resize(image, self.size)
        if self.flip:
            if self.prng.choice([True, False]):
                image = np.flip(image, axis=1)
        return image

    def get_example(self, i):
        choices = self.labels["choices"][i]
        if self.avoid_identity and len(choices) > 1:
            choices = [c for c in choices if c != i]
        j = self.prng.choice(choices)
        view0 = self.preprocess_image(self.labels["file_path_"][i])
        view1 = self.preprocess_image(self.labels["file_path_"][j])
        return {"view0": view0, "view1": view1}


class StochasticPairsWithMask(StochasticPairs):
    expected_data_csv_columns = [
        "character_id",
        "relative_file_path_",
        "relative_mask_path_",
    ]

    def __init__(self, config):
        """
        config: {
            "data_root": "foo",
            "data_csv": "train.csv",
            "spatial_size" : 256,
        }

        optional config parameters:
        config: {
            "mask_label" : 1,           # use mask label 1 for masking. can be a float. Default `1`.
            "invert_mask" : False,      # invert mask. This is useful if it is easier to just provide the background. Default `False`.
            "data_flip" : False,        # flip data randomly. Default `False`.
            "avoid_identity" : False,   # avoid the identity. Default `False`.
            "data_csv_columns" : ["character_id", "relative_file_path_"] # `list` of `str` column names or "from_csv",
            "data_csv_has_header" : False # default `False`
        }

        suggested data_csv layout
        id,relative_image_path_,col3,co4,...
        for example:
        1,frames/01/00001.jpg,xxx,yyy
        1,frames/01/00002.jpg,xxx,yyy
        ...
        2,frames/02/00001.jpg,xxx,yyy
        2,frames/02/00002.jpg,xxx,yyy

        If the csv has more columns, the other columns will be ignored.
        Parameters
        ----------
        config: dict with options. See above

        Examples
        --------
            See test
        """
        self.mask_label = config.get("mask_label", 1)
        self.invert_mask = config.get("invert_mask", False)
        super(StochasticPairsWithMask, self).__init__(config)

    def __len__(self):
        return self._length

    def preprocess_image(self, image_path: str, mask_path: str) -> np.ndarray:
        image = load_image(image_path)
        mask = load_image(mask_path)

        mask = mask == self.mask_label
        if self.invert_mask:
            mask = np.logical_not(mask)
        image = image * 1.0 * mask
        image = resize(image, self.size)
        if self.flip:
            if self.prng.choice([True, False]):
                image = np.flip(image, axis=1)
        return image

    def get_example(self, i) -> dict:
        choices = self.labels["choices"][i]
        if self.avoid_identity and len(choices) > 1:
            choices = [c for c in choices if c != i]
        j = self.prng.choice(choices)
        view0 = self.preprocess_image(
            self.labels["file_path_"][i], self.labels["mask_path_"][i]
        )
        view1 = self.preprocess_image(
            self.labels["file_path_"][j], self.labels["mask_path_"][j]
        )
        return {"view0": view0, "view1": view1}


class StochasticPairsWithSuperpixels(StochasticPairs):
    expected_data_csv_columns = ["character_id", "relative_file_path_"]

    def __init__(self, config):
        """
        config: {
            "data_root": "foo",
            "data_csv": "train.csv",
            "spatial_size" : 256,
        }

        optional config parameters:
        config: {
            "mask_label" : 1,           # use mask label 1 for masking. can be a float. Default `1`.
            "invert_mask" : False,      # invert mask. This is useful if it is easier to just provide the background. Default `False`.
            "data_flip" : False,        # flip data randomly. Default `False`.
            "avoid_identity" : False,   # avoid the identity. Default `False`.
            "data_csv_columns" : ["character_id", "relative_file_path_"] # `list` of `str` column names or "from_csv",
            "data_csv_has_header" : False # default `False`,
            "superpixel_params": {
                "n_segments" : 250,
                "compactness" : 10,
                "sigma" : 1
            }  # default values
        }

        suggested data_csv layout
        id,relative_image_path_,col3,co4,...
        for example:
        1,frames/01/00001.jpg,xxx,yyy
        1,frames/01/00002.jpg,xxx,yyy
        ...
        2,frames/02/00001.jpg,xxx,yyy
        2,frames/02/00002.jpg,xxx,yyy

        If the csv has more columns, the other columns will be ignored.
        Parameters
        ----------
        config: dict with options. See above

        Examples
        --------
            See test
        """
        super(StochasticPairsWithSuperpixels, self).__init__(config)
        default_superpixel_params = {"n_segments": 250, "compactness": 10, "sigma": 1}
        self.superpixel_params = config.get(
            "superpixel_params", default_superpixel_params
        )

    def __len__(self):
        return self._length

    def preprocess_image(self, image_path: str) -> np.ndarray:
        image = load_image(image_path)
        image = resize(image, self.size)
        return image

    def get_example(self, i):
        choices = self.labels["choices"][i]
        if self.avoid_identity and len(choices) > 1:
            choices = [c for c in choices if c != i]
        j = self.prng.choice(choices)

        view0 = self.preprocess_image(self.labels["file_path_"][i])
        view1 = self.preprocess_image(self.labels["file_path_"][j])
        superpixel_segments0 = slic(view0, **self.superpixel_params)
        superpixel_segments0 = edu.resize_labels(
            superpixel_segments0, (self.size, self.size)
        )
        superpixel_segments1 = slic(view1, **self.superpixel_params)
        superpixel_segments1 = edu.resize_labels(
            superpixel_segments1, (self.size, self.size)
        )
        superpixel_segments0 = np.expand_dims(superpixel_segments0, -1)
        superpixel_segments1 = np.expand_dims(superpixel_segments1, -1)
        if self.flip:
            if self.prng.choice([True, False]):
                view0 = np.flip(view0, axis=1)
                superpixel_segments0 = np.flip(superpixel_segments0, axis=1)

        if self.flip:
            if self.prng.choice([True, False]):
                view1 = np.flip(view1, axis=1)
                superpixel_segments1 = np.flip(superpixel_segments1, axis=1)
        return {
            "view0": view0,
            "view1": view1,
            "segments0": superpixel_segments0.astype(np.int32),
            "segments1": superpixel_segments1.astype(np.int32),
        }


class StochasticPairsWithMaskWithSuperpixels(StochasticPairsWithMask):
    expected_data_csv_columns = [
        "character_id",
        "relative_file_path_",
        "relative_mask_path_",
    ]

    def __init__(self, config):
        super(StochasticPairsWithMaskWithSuperpixels, self).__init__(config)
        default_superpixel_params = {"n_segments": 250, "compactness": 10, "sigma": 1}
        self.superpixel_params = config.get(
            "superpixel_params", default_superpixel_params
        )

    def __len__(self):
        return self._length

    def preprocess_image(self, image_path: str, mask_path: str) -> np.ndarray:
        image = load_image(image_path)
        mask = load_image(mask_path)

        mask = mask == self.mask_label
        if self.invert_mask:
            mask = np.logical_not(mask)
        image = image * 1.0 * mask
        image = resize(image, self.size)
        return image

    def get_example(self, i):
        choices = self.labels["choices"][i]
        if self.avoid_identity and len(choices) > 1:
            choices = [c for c in choices if c != i]
        j = self.prng.choice(choices)

        view0 = self.preprocess_image(
            self.labels["file_path_"][i], self.labels["mask_path_"][i]
        )
        view1 = self.preprocess_image(
            self.labels["file_path_"][j], self.labels["mask_path_"][j]
        )
        superpixel_segments0 = slic(view0, **self.superpixel_params)
        superpixel_segments0 = edu.resize_labels(
            superpixel_segments0, (self.size, self.size)
        )
        superpixel_segments1 = slic(view1, **self.superpixel_params)
        superpixel_segments1 = edu.resize_labels(
            superpixel_segments1, (self.size, self.size)
        )
        superpixel_segments0 = np.expand_dims(superpixel_segments0, -1)
        superpixel_segments1 = np.expand_dims(superpixel_segments1, -1)
        if self.flip:
            if self.prng.choice([True, False]):
                view0 = np.flip(view0, axis=1)
                superpixel_segments0 = np.flip(superpixel_segments0, axis=1)

        if self.flip:
            if self.prng.choice([True, False]):
                view1 = np.flip(view1, axis=1)
                superpixel_segments1 = np.flip(superpixel_segments1, axis=1)
        example = {
            "view0": view0,
            "view1": view1,
            "segments0": superpixel_segments0.astype(np.int32),
            "segments1": superpixel_segments1.astype(np.int32),
        }
        return example
