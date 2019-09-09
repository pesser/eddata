import pytest
from eddata import stochastic_pair
import os
import pandas as pd


def setup_csv_with_masks(tmpdir):
    data = {
        "id": [1] * 10,
        "image_path": ["im.png"] * 10,
        "mask_path": ["mask.png"] * 10,
    }
    df = pd.DataFrame(data)
    df.to_csv(os.path.join(tmpdir, "data.csv"), index=False, header=False)


def setup_csv_with_segments(tmpdir):
    data = {
        "id": [1] * 10,
        "image_path": ["im.png"] * 10,
        "mask_path": ["mask.png"] * 10,
        "segment_path": ["segment.png"] * 10,
    }
    df = pd.DataFrame(data)
    df.to_csv(os.path.join(tmpdir, "data.csv"), index=False, header=False)


def setup_csv_with_images(tmpdir):
    data = {"id": [1] * 10, "image_path": ["im.png"] * 10}
    df = pd.DataFrame(data)
    df.to_csv(os.path.join(tmpdir, "data.csv"), index=False, header=False)


def setup_csv_with_many_columns(tmpdir):
    data = {
        "id": [1] * 10,
        "image_path": ["im.png"] * 10,
        "mask_path": ["mask.png"] * 10,
        "foo": ["bar"] * 10,
    }
    df = pd.DataFrame(data)
    df.to_csv(os.path.join(tmpdir, "data.csv"), index=False, header=False)


class Test_StochasticPairs(object):
    def test_csv_has_more_columns(self, tmpdir):
        p = tmpdir.mkdir("data")
        setup_csv_with_masks(p)
        config = {
            "data_root": p,
            "data_csv": os.path.join(p, "data.csv"),
            "spatial_size": (256, 256),
        }
        dset = stochastic_pair.StochasticPairs(config)
        unique_labels = set(dset.labels["character_id"])
        assert set([1]) == unique_labels

        unique_file_paths = set(dset.labels["relative_file_path_"])
        assert set(["im.png"]) == unique_file_paths

        unique_file_paths = set(dset.labels["file_path_"])
        assert set([os.path.join(p, "im.png")]) == unique_file_paths

    def test_load_headers_from_csv(self, tmpdir):
        """
        config = {
            "data_root": "/mnt/comp/code/nips19/data/exercise_data/exercise_dataset/",
            "data_csv": "/mnt/comp/code/nips19/data/exercise_data/exercise_dataset/csvs/instance_level_train_split.csv",
            "data_avoid_identity": False,
            "data_flip": True,
            "spatial_size" : 256,
            "mask_label" : 255,
            "invert_mask" : False,
            "data_csv_columns" : "from_csv"
        }
        Parameters
        ----------
        tmpdir

        Returns
        -------
        """

        def setup_csv(tmpdir):
            data = {"character_id": [1] * 10, "relative_file_path_": ["im.png"] * 10}
            df = pd.DataFrame(data)
            df.to_csv(os.path.join(tmpdir, "data.csv"), index=False, header=True)

        p = tmpdir.mkdir("data")
        setup_csv(p)
        config = {
            "data_root": p,
            "data_csv": os.path.join(p, "data.csv"),
            "spatial_size": (256, 256),
            "data_csv_columns": "from_csv",
            "data_csv_has_header": True,
        }
        dset = stochastic_pair.StochasticPairs(config)

        dset_labels_set = {*dset.labels.keys()}
        assert all(
            [len(dset.labels[k]) == 10 for k in dset_labels_set.difference({"choices"})]
        )

        unique_labels = set(dset.labels["character_id"])
        assert set([1]) == unique_labels

        unique_file_paths = set(dset.labels["relative_file_path_"])
        assert set(["im.png"]) == unique_file_paths

        unique_file_paths = set(dset.labels["file_path_"])
        assert set([os.path.join(p, "im.png")]) == unique_file_paths

    def test_provide_headers_in_config(self, tmpdir):
        """
        config = {
            "data_root": "/mnt/comp/code/nips19/data/exercise_data/exercise_dataset/",
            "data_csv": "/mnt/comp/code/nips19/data/exercise_data/exercise_dataset/csvs/instance_level_train_split.csv",
            "data_avoid_identity": False,
            "data_flip": True,
            "spatial_size" : 256,
            "mask_label" : 255,
            "invert_mask" : False,
            "data_csv_header" : ["character_id", "relative_file_path_"]
            "data
        }
        Parameters
        ----------
        tmpdir

        Returns
        -------
        """

        def setup_csv(tmpdir):
            data = {"id": [1] * 10, "im_path": ["im.png"] * 10}
            df = pd.DataFrame(data)
            df.to_csv(os.path.join(tmpdir, "data.csv"), index=False, header=True)

        p = tmpdir.mkdir("data")
        setup_csv(p)
        config = {
            "data_root": p,
            "data_csv": os.path.join(p, "data.csv"),
            "spatial_size": (256, 256),
            "data_csv_columns": ["character_id", "relative_file_path_"],
            "data_csv_has_header": True,
        }
        dset = stochastic_pair.StochasticPairs(config)

        dset_labels_set = {*dset.labels.keys()}
        assert all(
            [len(dset.labels[k]) == 10 for k in dset_labels_set.difference({"choices"})]
        )
        unique_labels = set(dset.labels["character_id"])
        assert unique_labels == set([1])

        unique_file_paths = set(dset.labels["relative_file_path_"])
        assert unique_file_paths == set(["im.png"])

        unique_file_paths = set(dset.labels["file_path_"])
        assert unique_file_paths == set([os.path.join(p, "im.png")])

    def test_provide_headers_in_config2(self, tmpdir):
        """
        config = {
            "data_root": "/mnt/comp/code/nips19/data/exercise_data/exercise_dataset/",
            "data_csv": "/mnt/comp/code/nips19/data/exercise_data/exercise_dataset/csvs/instance_level_train_split.csv",
            "data_avoid_identity": False,
            "data_flip": True,
            "spatial_size" : 256,
            "mask_label" : 255,
            "invert_mask" : False,
            "data_csv_columns" : ["character_id", "relative_file_path_"]
            "data
        }
        Parameters
        ----------
        tmpdir

        Returns
        -------
        """

        def setup_csv(tmpdir):
            data = {"id": [1] * 10, "im_path": ["im.png"] * 10}
            df = pd.DataFrame(data)
            df.to_csv(os.path.join(tmpdir, "data.csv"), index=False, header=False)

        p = tmpdir.mkdir("data")
        setup_csv(p)
        config = {
            "data_root": p,
            "data_csv": os.path.join(p, "data.csv"),
            "spatial_size": (256, 256),
            "data_csv_header": ["character_id", "relative_file_path_"],
            "data_csv_has_header": False,
        }
        dset = stochastic_pair.StochasticPairs(config)

        dset_labels_set = {*dset.labels.keys()}
        assert all(
            [len(dset.labels[k]) == 10 for k in dset_labels_set.difference({"choices"})]
        )
        unique_labels = set(dset.labels["character_id"])
        assert set([1]) == unique_labels

        unique_file_paths = set(dset.labels["relative_file_path_"])
        assert set(["im.png"]) == unique_file_paths

        unique_file_paths = set(dset.labels["file_path_"])
        assert set([os.path.join(p, "im.png")]) == unique_file_paths


class Test_StochasticPairsWithMask(object):
    def test_load_csv(self, tmpdir):
        p = tmpdir.mkdir("data")
        setup_csv_with_masks(p)
        config = {
            "data_root": p,
            "data_csv": os.path.join(p, "data.csv"),
            "spatial_size": (256, 256),
        }
        dset = stochastic_pair.StochasticPairsWithMask(config)
        unique_labels = set(dset.labels["character_id"])
        assert set(["1"]) == unique_labels

        unique_file_paths = set(dset.labels["relative_file_path_"])
        assert set(["im.png"]) == unique_file_paths

        unique_file_paths = set(dset.labels["file_path_"])
        assert set([os.path.join(p, "im.png")]) == unique_file_paths

        unique_mask_paths = set(dset.labels["relative_mask_path_"])
        assert set(["mask.png"]) == unique_mask_paths

        unique_mask_paths = set(dset.labels["mask_path_"])
        assert set([os.path.join(p, "mask.png")]) == unique_mask_paths

    def test_csv_has_more_columns(self, tmpdir):
        p = tmpdir.mkdir("data")
        setup_csv_with_many_columns(p)
        config = {
            "data_root": p,
            "data_csv": os.path.join(p, "data.csv"),
            "spatial_size": (256, 256),
        }
        dset = stochastic_pair.StochasticPairsWithMask(config)
        unique_labels = set(dset.labels["character_id"])
        assert set(["1"]) == unique_labels

        unique_file_paths = set(dset.labels["relative_file_path_"])
        assert set(["im.png"]) == unique_file_paths

        unique_mask_paths = set(dset.labels["relative_mask_path_"])
        assert set(["mask.png"]) == unique_mask_paths


# TODO: parameterize test_load_csv to avoid dublicate code between Stochastic pair loaders


class Test_StochasticPairsWithSuperpixels(object):
    def test_load_csv(self, tmpdir):
        p = tmpdir.mkdir("data")
        setup_csv_with_segments(p)
        config = {
            "data_root": p,
            "data_csv": os.path.join(p, "data.csv"),
            "spatial_size": (256, 256),
        }
        dset = stochastic_pair.StochasticPairsWithSuperpixels(config)
        unique_labels = set(dset.labels["character_id"])
        assert set(["1"]) == unique_labels

        unique_file_paths = set(dset.labels["relative_file_path_"])
        assert set(["im.png"]) == unique_file_paths

        unique_file_paths = set(dset.labels["file_path_"])
        assert set([os.path.join(p, "im.png")]) == unique_file_paths

        unique_mask_paths = set(dset.labels["relative_mask_path_"])
        assert set(["mask.png"]) == unique_mask_paths

        unique_mask_paths = set(dset.labels["mask_path_"])
        assert set([os.path.join(p, "mask.png")]) == unique_mask_paths

        unique_segment_paths = set(dset.labels["relative_segment_path_"])
        assert set(["segment.png"]) == unique_segment_paths

        unique_segment_paths = set(dset.labels["segment_path_"])
        assert set([os.path.join(p, "segment.png")]) == unique_segment_paths
