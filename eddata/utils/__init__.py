import os
from pathlib import Path
import numpy as np
from tqdm import tqdm
import urllib
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


def resize_labels(labels, size):
    """Reshape labels image to target size.

    Parameters
    ----------
    labels : np.ndarray
        [H, W] or [N, H, W] - shaped array where each pixel is an `int` giving a label id for the segmentation. In case of [N, H, W],
        each slice along the first dimension is treated as an independent label image.
    size : tuple of ints
        Target shape as tuple of ints

    Returns
    -------
    reshaped_labels : np.ndarray
        [size[0], size[1]] or [N, size[0], size[1]]-shaped array

    Raises
    ------
    ValueError
        if labels does not have valid shape
    """
    # TODO: make this work for a single image
    if len(labels.shape) == 2:
        return cv2.resize(labels, size, 0, 0, cv2.INTER_NEAREST)
    elif len(labels.shape) == 3:
        label_list = np.split(labels, labels.shape[0], axis=0)
        label_list = list(
            map(
                lambda x: cv2.resize(np.squeeze(x), size, 0, 0, cv2.INTER_NEAREST),
                label_list,
            )
        )
        labels = np.stack(label_list, axis=0)
        return labels
    else:
        raise ValueError("unsupported shape for labels : {}".format(labels.shape))
