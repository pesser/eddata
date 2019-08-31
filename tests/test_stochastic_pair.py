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
        assert set(["1"]) == unique_labels

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
