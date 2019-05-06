import sys, os, tarfile, pickle
from pathlib import Path
import numpy as np
import eddata.utils as edu
from scipy.io import loadmat
from tqdm import tqdm


class PennAction(edu.DatasetMixin):
    def __init__(self, config = None):
        self.config = config or dict()
        self._prepare()
        self._load()

    def _prepare(self):
        self.root = edu.get_root("PennAction")
        self._label_path = Path(self.root).joinpath("eddata_labels.p")
        if not edu.is_prepared(self.root):
            # prep
            root = Path(self.root)
            if not root.joinpath("Penn_Action").is_dir():
                tarpath = edu.prompt_download(
                        file_ = "Penn_Action.tar.gz",
                        source = "https://upenn.app.box.com/v/PennAction",
                        target_dir = root)
                print("Extracting {}.".format(tarpath))
                with tarfile.open(tarpath, "r") as f:
                    f.extractall(path = root)
                print("Done extracting.")

            print("Generating labels.")
            label_keys = ['action', 'bbox', 'dimensions', 'nframes', 'pose',
                          'train', 'visibility', 'x', 'y', 'video_id', 'image_path']
            labels = dict((k, list()) for k in label_keys)

            label_dir = root.joinpath("Penn_Action", "labels")
            video_ids = sorted(root.joinpath("Penn_Action", "frames").iterdir())
            exclude = set(["1154", "1865"]) # missing bounding boxes
            for video_id in tqdm(video_ids):
                if video_id.name in exclude:
                    continue
                frames = sorted(video_id.glob("*.jpg"))
                label_file = label_dir.joinpath(video_id.name+".mat")
                label_file = loadmat(label_file)
                frame_labels = dict()
                for i, frame_path in enumerate(frames):
                    frame_labels = {
                            "action": str(label_file["action"][0]),
                            "bbox": np.array(label_file["bbox"][i]),
                            "dimensions": np.array(label_file["dimensions"][0]),
                            "nframes": label_file["nframes"][0][0],
                            "pose": str(label_file["pose"][0]),
                            "train": label_file["train"][0][0] == 1,
                            "visibility": label_file["visibility"][i],
                            "x": label_file["x"][i],
                            "y": label_file["y"][i]}
                    frame_labels["video_id"] = video_id.name
                    frame_labels["image_path"] = str(frame_path.relative_to(root))
                    for k in label_keys:
                        labels[k].append(frame_labels[k])

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
        return example


class PennActionCropped(PennAction):
    def get_example(self, i):
        example = super().get_example(i)
        image = edu.load_image(os.path.join(self.root, example["image_path"]))
        example["image"] = edu.quadratic_crop(image, example["bbox"], alpha = 1.0)
        example["image"] = edu.resize_float32(example["image"], self.config.get("spatial_size", 256))
        return example
