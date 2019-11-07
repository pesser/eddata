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
            "data_flip_h": False,       # flip data randomly in horizontal direction
            "data_flip_v": False,       # flip data randomly in vertical direction
            "data_rotate": False,       # rotate data randomly in
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

        if config.get("data_flip", False):
            import warnings

            # TODO: this warning actually does not work

            warnings.warn(
                "use 'data_flip_h', 'data_flip_v' instead of data_flip",
                DeprecationWarning,
            )
            self.flip_h = self.flip_v = config.get("data_flip")
        else:
            self.flip_h = config.get("data_flip_h", False)
            self.flip_v = config.get("data_flip_v", False)
        self.rotate = config.get("data_rotate", False)
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
        self._length = len(list(self.labels.values())[0])

    def __len__(self):
        return self._length

    def preprocess_image(self, image_path):
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

        view0, = self.augment_data(view0)
        view1, = self.augment_data(view1)

        return {"view0": view0, "view1": view1}

    def augment_data(self, *images):
        if self.flip_h:
            images = self.stochastic_flip_h(*images)
        if self.flip_v:
            images = self.stochastic_flip_v(*images)
        if self.rotate:
            images = self.stochastic_rotate(*images)
        return images

    def stochastic_flip_h(self, *images):
        if self.prng.choice([True, False]):
            images = [np.flip(i, axis=1) for i in images]
        return images

    def stochastic_flip_v(self, *images):
        if self.prng.choice([True, False]):
            images = [np.flip(i, axis=0) for i in images]
        return images

    def stochastic_rotate(self, *images):
        how_many_rotations = self.prng.choice([0, 1, 2, 3])
        images = [np.rot90(i, how_many_rotations) for i in images]
        return images


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
            "data_flip_h": False,       # flip data randomly in horizontal direction
            "data_flip_v": False,       # flip data randomly in vertical direction
            "data_rotate": False,       # rotate data randomly in
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
        self.apply_mask = config.get("apply_mask", True)
        super(StochasticPairsWithMask, self).__init__(config)

    def __len__(self):
        return self._length

    def preprocess_image(self, image_path: str, mask_path: str) -> np.ndarray:
        image = load_image(image_path)
        mask = load_image(mask_path)

        mask = mask == self.mask_label
        if self.invert_mask:
            mask = np.logical_not(mask)
        if self.apply_mask:
            image = image * 1.0 * mask
            image = resize(image, self.size)
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

        view0, = self.augment_data(view0)
        view1, = self.augment_data(view1)
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
            "data_flip_h": False,       # flip data randomly in horizontal direction
            "data_flip_v": False,       # flip data randomly in vertical direction
            "data_rotate": False,       # rotate data randomly in
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
        superpixel_segments0, view0 = self.augment_data(superpixel_segments0, view0)
        superpixel_segments1, view1 = self.augment_data(superpixel_segments1, view1)

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

        view0, superpixel_segments0 = self.augment_data(view0, superpixel_segments0)
        view1, superpixel_segments1 = self.augment_data(view1, superpixel_segments1)

        example = {
            "view0": view0,
            "view1": view1,
            "segments0": superpixel_segments0.astype(np.int32),
            "segments1": superpixel_segments1.astype(np.int32),
        }
        return example


if __name__ == "__main__":
    config = {
        "data_root": "/mnt/comp/code/nips19/data/exercise_data/exercise_dataset/",
        "data_csv": "/mnt/comp/code/nips19/data/exercise_data/exercise_dataset/csvs/instance_level_train_split.csv",
        "data_avoid_identity": False,
        "data_flip": True,
        "spatial_size": 256,
        "mask_label": 255,
        "invert_mask": False,
        "data_csv_columns": ["character_id", "relative_file_path_"],
        "data_csv_has_header": True,
    }
    dset = StochasticPairs(config)
    example = dset.get_example(4)
    view0 = example["view0"]
    view1 = example["view1"]
