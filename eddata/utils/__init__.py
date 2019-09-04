import os
from pathlib import Path
import numpy as np
from tqdm import tqdm
import urllib
import pandas as pd
from edflow.iterators.batches import (
    load_image,
    DatasetMixin,
    resize_float32,
    save_image,
)


def reporthook(bar):
    """tqdm progress bar for downloads."""

    def hook(b=1, bsize=1, tsize=None):
        if tsize is not None:
            bar.total = tsize
        bar.update(b * bsize - bar.n)

    return hook


def get_root(name):
    base = os.environ.get("EDDATA_CACHE", os.path.expanduser("~/.eddata_cache"))
    root = os.path.join(base, name)
    os.makedirs(root, exist_ok=True)
    return root


def is_prepared(root):
    return Path(root).joinpath(".ready").exists()


def mark_prepared(root):
    Path(root).joinpath(".ready").touch()


def prompt_download(file_, source, target_dir):
    targetpath = os.path.join(target_dir, file_)
    while not os.path.exists(targetpath):
        print(
            "Please download '{}' from '{}' to '{}'.".format(file_, source, targetpath)
        )
        input("Press Enter when done...")
    return targetpath


def download_url(file_, url, target_dir):
    targetpath = os.path.join(target_dir, file_)
    os.makedirs(target_dir, exist_ok=True)
    with tqdm(
        unit="B", unit_scale=True, unit_divisor=1024, miniters=1, desc=file_
    ) as bar:
        urllib.request.urlretrieve(url, targetpath, reporthook=reporthook(bar))
    return targetpath


def download_urls(urls, target_dir):
    paths = dict()
    for fname, url in urls.items():
        outpath = download_url(fname, url, target_dir)
        paths[fname] = outpath
    return paths


def quadratic_crop(x, bbox, alpha=1.0):
    """bbox is xmin, ymin, xmax, ymax"""
    im_h, im_w = x.shape[:2]
    bbox = np.array(bbox, dtype=np.float32)
    bbox = np.clip(bbox, 0, max(im_h, im_w))
    center = 0.5 * (bbox[0] + bbox[2]), 0.5 * (bbox[1] + bbox[3])
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    l = int(alpha * max(w, h))
    l = max(l, 2)

    required_padding = -1 * min(
        center[0] - l, center[1] - l, im_w - (center[0] + l), im_h - (center[1] + l)
    )
    required_padding = int(np.ceil(required_padding))
    if required_padding > 0:
        padding = [
            [required_padding, required_padding],
            [required_padding, required_padding],
        ]
        padding += [[0, 0]] * (len(x.shape) - 2)
        x = np.pad(x, padding, "reflect")
        center = center[0] + required_padding, center[1] + required_padding
    xmin = int(center[0] - l / 2)
    ymin = int(center[1] - l / 2)
    return np.array(x[ymin : ymin + l, xmin : xmin + l, ...])


def listify_dict(dict_):
    """Takes a dictionary and replaces all non-list values with a list
    with same length as the rest of the list values.
    If not list is present in the dict, every value will be turned
    into a list of length 1.
    This function is useful to convert dicts in pandas dataframe compatible dictionaries

    Parameters
    ----------
    dict_ : dict
        dictionary with list and non-list elements

    Returns
    -------
    dict
        dictionary with new elements. Lists are copied from input dict_.
        Non-list elements are replaced with lists of same length as list elements
        containint the non-list elements duplicated as often as the length of the list elements.

    Raises
    ------
    ValueError
        if list values are not of equal length

    Examples
    --------

    dict_ = {1 : 2, 3 : [4, 5]}
    pandaslib.listify_dict(dict_)
    >>> {1: [2, 2], 3: [4, 5]}

    dict_ = {1 : 2, 3: 4}
    pandaslib.listify_dict(dict_)
    >>> {1 : [2], 3 : [4]}
    """

    def len_if_list(x):
        if isinstance(x, list):
            return len(x)

    values = dict_.values()
    list_lengths = list(map(len_if_list, values))
    list_lengths = list(filter(lambda x: x is not None, list_lengths))
    list_lengths = set(list_lengths)
    if not len(list_lengths):
        # dict does not contain any list
        list_lengths.add(1)
    elif not len(list_lengths) == 1:
        raise ValueError("list values are not of equal length")
    final_list_length = list_lengths.pop()
    new_dict = {}
    for key, value in dict_.items():
        if isinstance(value, list):
            new_dict[key] = value
        else:
            new_dict[key] = [value] * final_list_length
    return new_dict


def add_abs_paths(example, root):
    """add absolute paths by appending root too all "relative_" paths in example.
    Relative paths have the substring "relative_" at the beginning of the key.
    """
    relative_path_keys = list(filter(lambda x: "relative_" in x, example.keys()))
    example_with_abspaths = {
        k.replace("relative_", ""): os.path.join(root, example[k])
        for k in relative_path_keys
    }
    example_with_abspaths.update(example)
    return example_with_abspaths


def df_empty(columns, dtypes=None, index=None):
    """create empty dataframe from column names and specified dtypes

    Parameters
    ----------
    columns : list
        list of str specifying column names
    dtypes : list of dtypes, optional
        list of dtypes for each column
    index : bool, optional
        [description], by default None

    Returns
    -------
    df
        empty pandas dataframe

    Examples
    --------
        df = df_empty(['a', 'b'], dtypes=[np.int64, np.int64])
        print(list(df.dtypes)) # int64, int64

        df = df_empty(['a', 'b'], dtypes=None)
        print(list(df.dtypes)) # float64, float64

    References
    ----------
        Shamelessly copied from https://stackoverflow.com/questions/36462257/create-empty-dataframe-in-pandas-specifying-column-types
    """
    if dtypes is None:
        dtypes = [None] * len(columns)
    has_consistent_lengths = len(columns) == len(dtypes)
    if not has_consistent_lengths:
        raise ValueError("columns and dtypes have to have same length")
    df = pd.DataFrame(index=index)
    for c, d in zip(columns, dtypes):
        df[c] = pd.Series(dtype=d)
    return df


def add_choices(labels, return_by_cid=False, character_id_key="character_id"):
    labels = dict(labels)
    cid_labels = np.asarray(labels[character_id_key])
    cids = np.unique(cid_labels)
    cid_indices = dict()
    for cid in cids:
        cid_indices[cid] = np.nonzero(cid_labels == cid)[0]
        verbose = False
        if verbose:
            if len(cid_indices[cid]) <= 1:
                print("No choice for {}: {}".format(cid, cid_indices[cid]))

    labels["choices"] = list()
    for i in range(len(labels[character_id_key])):
        cid = labels[character_id_key][i]
        choices = cid_indices[cid]
        labels["choices"].append(choices)
    if return_by_cid:
        return labels, cid_indices
    return labels
