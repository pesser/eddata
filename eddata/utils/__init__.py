import os
from pathlib import Path
from edflow.iterators.batches import load_image, DatasetMixin, resize_float32
import numpy as np


def get_root(name):
    base = os.environ.get("EDDATA_CACHE", os.path.expanduser("~/.eddata_cache"))
    root = os.path.join(base, name)
    os.makedirs(root, exist_ok = True)
    return root


def is_prepared(root):
    return Path(root).joinpath(".ready").exists()


def mark_prepared(root):
    Path(root).joinpath(".ready").touch()


def prompt_download(file_, source, target_dir):
    targetpath = os.path.join(target_dir, file_)
    while not os.path.exists(targetpath):
        print("Please download '{}' from '{}' to '{}'.".format(file_, source, targetpath))
        input("Press Enter when done...")
    return targetpath


def quadratic_crop(x, bbox, alpha = 1.0):
    """bbox is xmin, ymin, xmax, ymax"""
    center = 0.5*(bbox[0]+bbox[2]), 0.5*(bbox[1]+bbox[3])
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    l = int(alpha*max(w, h))

    im_h, im_w = x.shape[:2]
    required_padding = -1*min(
            center[0] - l,
            center[1] - l,
            im_w - (center[0] + l),
            im_h - (center[1] + l))
    required_padding = int(np.ceil(required_padding))
    if required_padding > 0:
        padding = [[required_padding, required_padding], [required_padding, required_padding]]
        padding += [[0,0]]*(len(x.shape)-2)
        x = np.pad(x, padding, "reflect")
        center = center[0]+required_padding, center[1]+required_padding
    xmin = int(center[0] - l/2)
    ymin = int(center[1] - l/2)
    return np.array(x[ymin:ymin+l,xmin:xmin+l,...])

