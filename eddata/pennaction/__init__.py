import sys, os, tarfile, pickle
from pathlib import Path
import numpy as np
import eddata.utils as edu
from scipy.io import loadmat
from tqdm import tqdm, trange
from matplotlib import pyplot as plt


class PennAction(edu.DatasetMixin):
    def __init__(self, config=None):
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
                    file_="Penn_Action.tar.gz",
                    source="https://upenn.app.box.com/v/PennAction",
                    target_dir=root,
                )
                print("Extracting {}.".format(tarpath))
                with tarfile.open(tarpath, "r") as f:
                    f.extractall(path=root)
                print("Done extracting.")

            print("Generating labels.")
            label_keys = [
                "action",
                "bbox",
                "dimensions",
                "nframes",
                "pose",
                "train",
                "visibility",
                "x",
                "y",
                "video_id",
                "image_path",
            ]
            labels = dict((k, list()) for k in label_keys)

            label_dir = root.joinpath("Penn_Action", "labels")
            video_ids = sorted(root.joinpath("Penn_Action", "frames").iterdir())
            exclude = set(["1154", "1865"])  # missing bounding boxes
            for video_id in tqdm(video_ids):
                if video_id.name in exclude:
                    continue
                frames = sorted(video_id.glob("*.jpg"))
                label_file = label_dir.joinpath(video_id.name + ".mat")
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
                        "y": label_file["y"][i],
                    }
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

    def __len__(self):
        return self._length


def preprocess_iuv(iuv_path):
    IUV = edu.load_image(iuv_path)
    I = IUV[:, :, 0]
    return I


class PennActionCropped(PennAction):
    def __init__(self, config=None):
        super().__init__(config)
        self._prepare_crops()

    def _prepare_crops(self):
        self.crop_root = Path(self.root).joinpath("Penn_Action", "cropped")
        self.crop_root.mkdir(exist_ok=True)
        if not edu.is_prepared(self.crop_root):
            frames_root = Path(self.root).joinpath("Penn_Action", "frames")
            csv = list()
            for i in trange(len(self)):
                image_path = self.labels["image_path"][i]
                video_id = self.labels["video_id"][i]
                image_path = Path(self.root).joinpath(image_path)
                sub_path = image_path.relative_to(frames_root)
                cropped_path = self.crop_root.joinpath(sub_path)

                if not cropped_path.exists():
                    cropped_example = self._get_cropped_example(i)
                    cropped_image = cropped_example["image"]
                    cropped_path.parent.mkdir(exist_ok=True)
                    edu.save_image(cropped_image, cropped_path)

                rel_cropped_path = cropped_path.relative_to(self.root)
                csv.append("{},{}".format(video_id, rel_cropped_path))

            csv_path = Path(self.root).joinpath("cropped.csv")
            with open(csv_path, "w") as f:
                f.write("\n".join(csv) + "\n")
            csv_train = [csv[i] for i in range(len(self)) if self.labels["train"][i]]
            with open(Path(self.root).joinpath("cropped_train.csv"), "w") as f:
                f.write("\n".join(csv_train) + "\n")
            csv_test = [csv[i] for i in range(len(self)) if not self.labels["train"][i]]
            with open(Path(self.root).joinpath("cropped_test.csv"), "w") as f:
                f.write("\n".join(csv_test) + "\n")

            for a in [
                "baseball_pitch",
                "baseball_swing",
                "bench_press",
                "bowl",
                "clean_and_jerk",
                "golf_swing",
                "jump_rope",
                "jumping_jacks",
                "pullup",
                "pushup",
                "situp",
                "squat",
                "strum_guitar",
                "tennis_forehand",
                "tennis_serve",
            ]:
                # all
                action_indices = [
                    i for i in range(len(self)) if self.labels["action"][i] == a
                ]
                action_csv = [csv[i] for i in action_indices]
                csv_path = Path(self.root).joinpath("cropped_" + a + ".csv")
                with open(csv_path, "w") as f:
                    f.write("\n".join(action_csv) + "\n")

                # train
                action_indices = [
                    i
                    for i in range(len(self))
                    if self.labels["action"][i] == a and self.labels["train"][i]
                ]
                action_csv = [csv[i] for i in action_indices]
                csv_path = Path(self.root).joinpath("cropped_" + a + "_train.csv")
                with open(csv_path, "w") as f:
                    f.write("\n".join(action_csv) + "\n")

                # test
                action_indices = [
                    i
                    for i in range(len(self))
                    if self.labels["action"][i] == a and not self.labels["train"][i]
                ]
                action_csv = [csv[i] for i in action_indices]
                csv_path = Path(self.root).joinpath("cropped_" + a + "_test.csv")
                with open(csv_path, "w") as f:
                    f.write("\n".join(action_csv) + "\n")
            edu.mark_prepared(self.crop_root)

    def _get_cropped_example(self, i):
        example = super().get_example(i)
        image = edu.load_image(os.path.join(self.root, example["image_path"]))
        example["image"] = edu.quadratic_crop(image, example["bbox"], alpha=1.0)
        return example

    def get_example(self, i):
        example = dict()
        for k in self.labels:
            example[k] = self.labels[k][i]
        cropped_path = example["image_path"].replace("frames", "cropped")
        example["image"] = edu.load_image(os.path.join(self.root, cropped_path))
        example["image"] = edu.resize_float32(
            example["image"], self.config.get("spatial_size", 256)
        )
        return example


class PennActionDenseposed(PennAction):
    def get_example(self, i):
        example = super(PennActionDenseposed, self).get_example(i)
        cropped_path = example["image_path"].replace("frames", "cropped")
        image = edu.load_image(os.path.join(self.root, cropped_path))
        mask_path = cropped_path.replace("cropped", "cropped_densepose")
        p, ext = os.path.splitext(mask_path)
        mask_path = "{}_IUV.png".format(p)
        mask = edu.load_image(os.path.join(self.root, mask_path))
        mask = mask[:, :, 0]
        image *= np.expand_dims(np.logical_not(mask == -1), -1)
        image = edu.resize_float32(image, self.config.get("spatial_size", 256))
        example["image"] = image
        return example


if __name__ == "__main__":
    d1 = PennAction()
    d2 = PennActionCropped()
    d3 = PennActionDenseposed()
    e = d3.get_example(0)["image"]
    plt.imshow(e)
    plt.savefig("e3.png")
