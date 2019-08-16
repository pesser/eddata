import sys, os, tarfile, pickle
from pathlib import Path
import numpy as np
import pandas as pd
import eddata.utils as edu
from scipy.io import loadmat
from tqdm import tqdm, trange


class DogsRunCropped(edu.DatasetMixin):
    def __init__(self, config=None):
        self.config = config or dict()
        self._prepare()
        self._load()

    def _extract_base_fname(self, dog_frame_fname):
        """cropped_dog001_01_0056.jpg --> dog001_01_0056.jpg"""
        base_dog_fname = dog_frame_fname.split("_")[1:]
        base_dog_fname = "_".join(base_dog_fname)
        return base_dog_fname

    def _list_dog_frames(self, base_path: str, dog_folder: str):
        """List dict of files within DogsRun/dogxxx_yy folder

        Parameters
        ----------
        base_path : XX/DogsRun
        dog_folder: dog001_01

        Returns : dict
        -------
            {"cropped : [dog001_01/cropped/*.jpg], "mask" : [dog001_01/mask/*.jpg] }
        """
        cropped_dog_frames = os.listdir(os.path.join(base_path, dog_folder, "cropped"))
        dog_basefnames = list(map(self._extract_base_fname, cropped_dog_frames))
        cropped_dog_fpaths = list(
            map(lambda x: "_".join(["cropped", x]), dog_basefnames)
        )
        cropped_dog_fpaths = list(
            map(
                lambda x: os.path.join("DogsRun", dog_folder, "cropped", x),
                cropped_dog_fpaths,
            )
        )
        mask_dog_fpaths = list(map(lambda x: "_".join(["mask", x]), dog_basefnames))
        mask_dog_fpaths = list(
            map(
                lambda x: os.path.join("DogsRun", dog_folder, "mask", x),
                mask_dog_fpaths,
            )
        )
        return {"cropped": sorted(cropped_dog_fpaths), "mask": sorted(mask_dog_fpaths)}

    def _extract_dog_id_from_folder(self, dog_folder):
        """Return dog Id from dog folder. The Dog id is the 3 digit number after the first 3 letters.

        Example : dog001_01 --> 1

        Parameters
        ----------
        dog_folder : str

        Returns
        -------
        dog Id as 'int'

        """
        d = dog_folder[3:6]
        return int(d)

    def _prepare(self):
        self.root = edu.get_root("DogsRun")
        self._label_path = Path(self.root).joinpath("eddata_labels.p")
        if not edu.is_prepared(self.root):
            root = Path(self.root)  # DogsRun
            if not root.joinpath("DogsRun").is_dir():  # DogsRun/DogsRun
                # TODO: implement download
                # tarpath = edu.prompt_download(
                #         file_ = "Penn_Action.tar.gz",
                #         source = "https://upenn.app.box.com/v/PennAction",
                #         target_dir = root)
                # print("Extracting {}.".format(tarpath))
                # with tarfile.open(tarpath, "r") as f:
                #     f.extractall(path = root)
                # print("Done extracting.")
                pass

            print("Generating labels.")
            base_path = root.joinpath("DogsRun")
            dogs_folders = sorted(os.listdir(base_path))
            data = []
            for dog_folder in dogs_folders:
                id_ = self._extract_dog_id_from_folder(dog_folder)
                frames = self._list_dog_frames(base_path, dog_folder)
                new_data = {"id": id_}
                new_data.update(frames)
                data.append(new_data)
            data_frame = pd.concat(map(pd.DataFrame, data))
            labels = data_frame.to_dict(orient="list")

            print("Saving labels.")
            with open(self._label_path, "wb") as f:
                pickle.dump(labels, f)

            edu.mark_prepared(self.root)

    def _load_labels(self):
        with open(self._label_path, "rb") as f:
            labels = pickle.load(f)
        return labels

    def _load(self):
        self.labels = self._load_labels()
        self._length = len(next(iter(self.labels.values())))

    def get_example(self, i):
        example = dict()
        for k in self.labels:
            example[k] = self.labels[k][i]
        cropped_path = example["cropped"]
        example["image"] = edu.load_image(os.path.join(self.root, cropped_path))
        example["image"] = edu.resize_float32(
            example["image"], self.config.get("spatial_size", 256)
        )
        return example

    def __len__(self):
        return self._length


class DogsRunMasked(DogsRunCropped):
    def get_example(self, i):
        example = super(DogsRunMasked, self).get_example(i)
        mask_path = example["mask"]
        mask = edu.load_image(os.path.join(self.root, mask_path))
        mask = edu.resize_float32(mask, self.config.get("spatial_size", 256))
        example["image"] = example["image"] * 1.0 * (mask == 1.0)
        example["image"] = edu.resize_float32(
            example["image"], self.config.get("spatial_size", 256)
        )
        return example


if __name__ == "__main__":
    from matplotlib import pyplot as plt

    d1 = DogsRunCropped()
    e = d1.get_example(0)["image"]
    plt.imshow(e)
    plt.savefig("e1.png")
    d2 = DogsRunMasked()
    e = d2.get_example(0)["image"]
    plt.imshow(e)
    plt.savefig("e2.png")
