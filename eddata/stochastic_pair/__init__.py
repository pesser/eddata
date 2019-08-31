from edflow.util import PRNGMixin
from edflow.iterators.batches import DatasetMixin
from edflow.iterators.batches import resize_float32 as resize
from edflow.iterators.batches import load_image
import os
import numpy as np


class StochasticPairs(DatasetMixin, PRNGMixin):
    def __init__(self, config):
        """config has to include
        config: {
            "data_root": "foo",
            "data_csv": "train.csv",
            "data_csv_header" : ["character_id", "relative_file_path_" : 1] # or "from_csv"
            "spatial_size" : 256,
        }

        optional config parameters:
        config: {
            "data_flip" : False,     # flip data randomly. Default `False`.
            "avoid_identity" : False, # avoid the identity. Default `False`.
        }

        data_csv has to have the following layout:
        id,image_path_from_data_root

        for example:
        1,frames/01/00001.jpg
        1,frames/01/00002.jpg
        ...
        2,frames/02/00001.jpg
        2,frames/02/00002.jpg

        If the csv has more columns, the other columns will be ignored.
        Parameters
        ----------
        config: dict with options. See above
        """
        self.config = config
        self.size = config["spatial_size"]
        self.root = config["data_root"]
        self.csv = config["data_csv"]
        self.avoid_identity = config.get("data_avoid_identity", True)
        self.flip = config.get("data_flip", False)
        self.make_labels()
        self.labels = add_choices(self.labels)

    def make_labels(self):
        expected_data_header = ["character_id", "relative_file_path_"]
        header = self.config.get("data_csv_header", expected_data_header)
        self.header = header  # type : list
        if header == "from_csv":
            raise NotImplementedError("from csv is not implemented yet")
        else:
            with open(self.csv) as f:
                lines = f.read().splitlines()
            lines = [l.split(",") for l in lines]

            self.labels = {
                label_name: [l[i] for l in lines]
                for label_name, i in zip(header, range(len(header)))
            }
            for label_name, i in zip(header, range(len(header))):
                if "file_path_" in label_name:
                    label_update = {
                        label_name.replace("relative_", ""): [
                            os.path.join(self.root, l[i]) for l in lines
                        ]
                    }
                    self.labels.update(label_update)
            self._length = len(lines)

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
    def __init__(self, config):
        """config has to include
        config: {
            "data_root": "foo",
            "data_csv": "train.csv",
            "spatial_size" : 256
        }

        optional config parameters:
        config: {
            "mask_label" : 1,        # use mask label 1 for masking. can be a float. Default `1`.
            "invert_mask" : False,   # invert mask. This is usefull if it is easier to just provide the background. Default `False`.
            "data_flip" : False,     # flip data randomly. Default `False`.
            "avoid_identity" : False, # avoid the identity. Default `False`.
        }

        data_csv has to have the following layout:
        id,image_path_from_data_root,mask_path_from_data_root

        for example:
        1,frames/01/00001.jpg,mask/01/0001.png
        1,frames/01/00002.jpg,mask/01/0002.png
        ...
        2,frames/02/00001.jpg,mask/02/0001.png
        2,frames/02/00002.jpg,mask/02/0002.png

        If the csv has more columns, the other columns will be ignored.
        Parameters
        ----------
        config: dict with options. See above
        """
        self.mask_label = config.get("mask_label", 1)
        self.invert_mask = config.get("invert_mask", False)
        super(StochasticPairsWithMask, self).__init__(config)

    def make_labels(self):
        expected_data_header = [
            "character_id",
            "relative_file_path_",
            "relative_mask_path_",
        ]
        header = self.config.get("data_csv_header", expected_data_header)
        if header == "from_csv":
            raise NotImplementedError("from csv is not implemented yet")
        else:
            with open(self.csv) as f:
                lines = f.read().splitlines()
            lines = [l.split(",") for l in lines]

            self.labels = {
                label_name: [l[i] for l in lines]
                for label_name, i in zip(header, range(len(header)))
            }
            for label_name, i in zip(header, range(len(header))):
                if "relative_" in label_name:
                    label_update = {
                        label_name.replace("relative_", ""): [
                            os.path.join(self.root, l[i]) for l in lines
                        ]
                    }
                    self.labels.update(label_update)
            self._length = len(lines)

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
    def __init__(self, config):
        """config has to include
                config: {
                    "data_root": "foo",
                    "data_csv": "train.csv",
                    "spatial_size" : 256
                    "data_labels" : "from_csv_header"
                }

                optional config parameters:
                config: {
                    "data_flip" : False,     # flip data randomly. Default `False`.
                    "avoid_identity" : False, # avoid the identity. Default `False`.
                }

                data_csv has to have the following layout:
                id,image_path_from_data_root,segment_path_from_data_root,segment_path_from_data_root

                for example:
                1,frames/01/00001.jpg,segments/01/0001.png
                1,frames/01/00002.jpg,segments/01/0002.png
                ...
                2,frames/02/00001.jpg,segments/02/0001.png
                2,frames/02/00002.jpg,segments/02/0002.png

                If the csv has more columns, the other columns will be ignored.
                Parameters
                ----------
                config: dict with options. See above
                """
        super(StochasticPairsWithSuperpixels, self).__init__(config)

    def make_labels(self):
        expected_data_header = [
            "character_id",
            "relative_file_path_",
            "relative_mask_path_",
            "relative_segment_path_",
        ]
        header = self.config.get("data_csv_header", expected_data_header)
        if header == "from_csv":
            raise NotImplementedError("from csv is not implemented yet")
        else:
            with open(self.csv) as f:
                lines = f.read().splitlines()
            lines = [l.split(",") for l in lines]

            self.labels = {
                label_name: [l[i] for l in lines]
                for label_name, i in zip(header, range(len(header)))
            }
            for label_name, i in zip(header, range(len(header))):
                if "relative_" in label_name:
                    label_update = {
                        label_name.replace("relative_", ""): [
                            os.path.join(self.root, l[i]) for l in lines
                        ]
                    }
                    self.labels.update(label_update)
            self._length = len(lines)

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
        if self.flip:
            if self.prng.choice([True, False]):
                view0 = np.flip(view0, axis=1)

        view1 = self.preprocess_image(self.labels["file_path_"][j])
        if self.flip:
            if self.prng.choice([True, False]):
                view1 = np.flip(view1, axis=1)

        superpixel_segments0 = self.preprocess_image(self.labels["segments_path_"][i])
        superpixel_segments1 = self.preprocess_image(self.labels["segments_path_"][j])

        return {
            "view0": view0,
            "view1": view1,
            "superpixel_segments0": superpixel_segments0,
            "superpixel_segments1": superpixel_segments1,
        }


class StochasticPairsWithMaskWithSuperpixels(StochasticPairsWithMask):
    def __init__(self, config):
        super(StochasticPairsWithMask, self).__init__(config)

    def __len__(self):
        return self._length

    def make_labels(self):
        expected_data_header = [
            "character_id",
            "relative_file_path_",
            "relative_mask_path_",
            "relative_segments_path_",
        ]
        header = self.config.get("data_csv_header", expected_data_header)
        if header == "from_csv":
            raise NotImplementedError("from csv is not implemented yet")
        else:
            with open(self.csv) as f:
                lines = f.read().splitlines()
            lines = [l.split(",")[:3] for l in lines]

            self.labels = {
                label_name: [l[i] for l in lines]
                for label_name, i in zip(header, range(len(header)))
            }
            for label_name, i in zip(header, range(len(header))):
                if "relative_" in label_name:
                    label_update = {
                        label_name.replace("relative_", ""): [
                            os.path.join(self.root, l[i]) for l in lines
                        ]
                    }
                    self.labels.update(label_update)
            self._length = len(lines)

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

        view0 = self.preprocess_image(self.labels["file_path_"][i])
        view1 = self.preprocess_image(self.labels["file_path_"][j])
        superpixel_segments0 = self.preprocess_image(self.labels["segments_path_"][i])
        superpixel_segments1 = self.preprocess_image(self.labels["segments_path_"][j])

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
            "superpixel_segments0": superpixel_segments0,
            "superpixel_segments1": superpixel_segments1,
        }
        return example


def add_choices(labels, return_by_cid=False):
    labels = dict(labels)
    cid_labels = np.asarray(labels["character_id"])
    cids = np.unique(cid_labels)
    cid_indices = dict()
    for cid in cids:
        cid_indices[cid] = np.nonzero(cid_labels == cid)[0]
        verbose = False
        if verbose:
            if len(cid_indices[cid]) <= 1:
                print("No choice for {}: {}".format(cid, cid_indices[cid]))

    labels["choices"] = list()
    for i in range(len(labels["character_id"])):
        cid = labels["character_id"][i]
        choices = cid_indices[cid]
        labels["choices"].append(choices)
    if return_by_cid:
        return labels, cid_indices
    return labels
