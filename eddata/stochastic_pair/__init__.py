from edflow.util import PRNGMixin
from edflow.iterators.batches import DatasetMixin
from edflow.iterators.batches import resize_float32 as resize
from edflow.iterators.batches import load_image
import os
import numpy as np


class StochasticPairsWithMask(DatasetMixin, PRNGMixin):
    def __init__(self, config):
        """

        config :{
            mask_label : 1 # use mask label 1 for masking. can be a float
            invert_mask : False # invert mask. This is usefull if it is easier to just provide the background
            data_flip : False # flip data randomly
        }

        Parameters
        ----------
        config: dict with options. See above
        """
        self.size = config["spatial_size"]
        self.root = config["data_root"]
        self.csv = config["data_csv"]
        self.avoid_identity = config.get("data_avoid_identity", True)
        self.flip = config.get("data_flip", False)
        self.mask_label = config.get("mask_label", 1)
        self.invert_mask = config.get("invert_mask", False)
        with open(self.csv) as f:
            lines = f.read().splitlines()
        self._length = len(lines)
        lines = [l.split(",", 3) for l in lines]
        self.labels = {
            "character_id": [l[0] for l in lines],
            "relative_file_path_": [l[1] for l in lines],
            "file_path_": [os.path.join(self.root, l[1]) for l in lines],
            "relative_mask_path_": [l[2] for l in lines],
            "mask_path_": [os.path.join(self.root, l[2]) for l in lines],
        }
        self.labels = add_choices(self.labels)

    def __len__(self):
        return self._length

    def preprocess_image(self, image_path, mask_path):
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
        return {"view0": view0, "view1": view1}


class StochasticPairs(DatasetMixin, PRNGMixin):
    def __init__(self, config):
        self.size = config["spatial_size"]
        self.root = config["data_root"]
        self.csv = config["data_csv"]
        self.avoid_identity = config.get("data_avoid_identity", True)
        self.flip = config.get("data_flip", False)
        with open(self.csv) as f:
            lines = f.read().splitlines()
        self._length = len(lines)
        lines = [l.split(",", 1) for l in lines]
        self.labels = {
            "character_id": [l[0] for l in lines],
            "relative_file_path_": [l[1] for l in lines],
            "file_path_": [os.path.join(self.root, l[1]) for l in lines],
        }
        self.labels = add_choices(self.labels)

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
