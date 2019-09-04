import pandas as pd
import os
from pathlib import Path
import numpy as np
import eddata.utils as edu
import traceback
import cv2

from eddata.utils import df_empty
from edflow.util import PRNGMixin


class VIP(edu.DatasetMixin):
    def __init__(self, config=None):
        # TODO: it looks like instance annotations are not consistent within videos. This is a problem for the stochastic pair case
        # for example video 17, frame 826 -> frame 851. One instance is leaving the frame here
        # but the annotation marks the instance remaining in the frame as instance 1 in 851 and instance 2 in 826
        # one would have to use reid manually on the instances to get the assignment within the video
        """

        References:
        ----------
        [1] http://sysu-hcp.net/lip/overview.php

            2.3 Video Multi-Person Human Parsing
            VIP(Video instance-level Parsing) dataset, the first video multi-person human parsing benchmark, consists of 404 videos covering various scenarios. For every 25 consecutive frames in each video, one frame is annotated densely with pixel-wise semantic part categories and instance-level identification. There are 21247 densely annotated images in total. We divide these 404 sequences into 304 train sequences, 50 validation sequences and 50 test sequences.

            You can also downlod this dataset at OneDrive and Baidu Drive.

            VIP_Fine: All annotated images and fine annotations for train and val sets.
            VIP_Sequence: 20-frame surrounding each VIP_Fine image (-10 | +10).
            VIP_Videos: 404 video sequences of VIP dataset.

        :param config:
        """
        self.config = config or dict()
        self.size = config.get("spatial_size", 256)
        self._prepare()
        self._load()

    def _prepare(self):
        self.root = edu.get_root("vip_fine")
        self._label_path = Path(self.root).joinpath("eddata_labels.csv")
        if not edu.is_prepared(self.root):
            # dataframe layout:
            # global_id | id_within_video | videoXXX | frame
            global_ids_csv = os.path.join(self.root, "global_ids.csv")
            if os.path.exists(global_ids_csv):
                columns = ["global_id", "id_within_video", "video", "frame"]
                dtypes = [np.int64, np.int64, str, str]
                dtypes = {k: v for k, v, in zip(columns, dtypes)}
                df_global_ids = pd.read_csv(global_ids_csv, dtype=dtypes)
            else:
                df_global_ids = self.make_global_human_ids()
                df_global_ids.to_csv(global_ids_csv, index=False)
            df_labels = df_empty(
                [
                    "id",
                    "id_within_video",
                    "relative_file_path_",
                    "relative_human_id_label_path_",
                ]
            )
            for (
                _,
                (global_id, id_within_video, video, frame),
            ) in df_global_ids.iterrows():
                new_data = {
                    "id": global_id,
                    "id_within_video": id_within_video,
                    "relative_file_path_": self.get_image_file(video, frame),
                    "relative_human_id_label_path_": self.get_human_instance_segmentation_file(
                        video, frame
                    ),
                }
                new_data = edu.listify_dict(new_data)
                new_df = pd.DataFrame(new_data)
                df_labels = pd.concat([df_labels, new_df], ignore_index=True)

            print("Saving labels.")
            df_labels.to_csv(self._label_path, index=False)

            edu.mark_prepared(self.root)

    def make_global_human_ids(self):
        base_path = os.path.join(self.root, "VIP_Fine")
        global_human_id = 0
        columns = ["global_id", "id_within_video", "video", "frame"]
        dtypes = [np.int64, np.int64, str, str]
        df = df_empty(columns, dtypes)
        videos = self.list_videos(base_path)
        failed_videos = []
        successfull_videos = []
        for video in videos:
            try:
                instance_id_annotation_files = self.get_instance_id_annotation_files(
                    base_path, video
                )
                human_ids_within_video = set([])
                for instance_id_annotation_file in instance_id_annotation_files:
                    human_ids_within_frame = self.get_human_ids_from_annotation_file(
                        instance_id_annotation_file
                    )
                    frame_number = self.get_frame_number_from_annotation_file(
                        instance_id_annotation_file
                    )
                    human_ids_within_video.update(set(human_ids_within_frame))
                    new_global_human_ids = list(
                        map(lambda v_id: v_id + global_human_id, human_ids_within_frame)
                    )
                    msg = "visiting {} - found ids {} - adding as {}".format(
                        video, human_ids_within_frame, new_global_human_ids
                    )
                    print(msg)
                    for new_global_human_id, new_id_within_frame in zip(
                        new_global_human_ids, human_ids_within_frame
                    ):
                        new_data_row = {
                            "global_id": new_global_human_id,
                            "id_within_video": new_id_within_frame,
                            "video": video,
                            "frame": frame_number,
                        }
                        df = df.append(new_data_row, ignore_index=True)
                global_human_id += max(human_ids_within_video)
                print(df.tail(4))
                successfull_videos.append(video)
            except Exception:
                traceback.print_exc()
                failed_videos.append(video)
        return df

    def _load_labels(self):
        print("loading labels")
        labels = pd.read_csv(self._label_path)
        return labels

    def _load(self):
        self.labels = self._load_labels()
        self._length = len(self.labels)

    def get_example(self, i):
        # example =
        # return example
        example = self.labels.iloc[i]
        example = dict(example)
        example = edu.add_abs_paths(example, self.root)
        return example

    def __len__(self):
        return self._length

    def list_image_files(self, video):
        relative_subpath = os.path.join("VIP_Fine", "Images", video)
        abs_path = os.path.join(self.root, relative_subpath)
        image_files = os.listdir(abs_path)
        image_files = sorted(image_files)
        image_files = list(
            map(lambda x: os.path.join(relative_subpath, x), image_files)
        )
        return image_files

    def list_human_instance_segmentation_files(self, video):
        relative_subpath = os.path.join("VIP_Fine", "Annotations", "Human_ids", video)
        abs_path = os.path.join(self.root, relative_subpath)
        image_files = os.listdir(abs_path)
        image_files = sorted(image_files)
        image_files = list(
            map(lambda x: os.path.join(relative_subpath, x), image_files)
        )
        return image_files

    def get_image_file(self, video: str, frame: str) -> str:
        relative_subpath = os.path.join("VIP_Fine", "Images", video)
        image_file = os.path.join(relative_subpath, frame + ".jpg")
        return image_file

    def get_human_instance_segmentation_file(self, video: str, frame: str) -> str:
        relative_subpath = os.path.join("VIP_Fine", "Annotations", "Human_ids", video)
        image_file = os.path.join(relative_subpath, frame + ".png")
        return image_file

    def get_instance_id_annotation_files(self, base_path, video):
        """get_instance_id_annotation_file("/mnt/hci_gpu/compvis_group/sabraun/LIP_VIP/VIP_Fine", "videos1")
        --> [
            /mnt/hci_gpu/compvis_group/sabraun/LIP_VIP/VIP_Fine/Annotations/Instance_ids/videos1/000000000001.txt,
            /mnt/hci_gpu/compvis_group/sabraun/LIP_VIP/VIP_Fine/Annotations/Instance_ids/videos1/000000000251.txt,
        ]
        Parameters
        ----------
        base_path
        video

        Returns
        -------

        """
        abs_path = os.path.join(base_path, "Annotations", "Instance_ids", video)
        txt_files = os.listdir(abs_path)
        txt_files = list(filter(lambda x: os.path.splitext(x)[1] == ".txt", txt_files))
        txt_files = sorted(txt_files)
        txt_files = list(map(lambda x: os.path.join(abs_path, x), txt_files))
        return txt_files

    def get_frame_number_from_annotation_file(self, annotation_file):
        """/mnt/hci_gpu/compvis_group/sabraun/LIP_VIP/VIP_Fine/Annotations/Instance_ids/videos1/000000000001.txt
        --> 000000000001"""
        fname = os.path.basename(annotation_file)
        fname_no_ext = os.path.splitext(fname)[0]
        return fname_no_ext

    def get_human_ids_from_annotation_file(self, instance_id_annotation_file):
        """/mnt/hci_gpu/compvis_group/sabraun/LIP_VIP/VIP_Fine/Annotations/Instance_ids/videos1/000000000001.txt
        --> [1, 2, 4]
        Parameters
        ----------
        instance_id_annotation_file

        Returns
        -------

        """
        ids = pd.read_csv(
            instance_id_annotation_file,
            sep=" ",
            header=None,
            names=["part_instance_id", "part_id", "human_id"],
        )
        human_ids = ids["human_id"]
        return np.unique(human_ids)

    def list_videos(self, base_path):
        """list_videos("xxx/LIP_VIP/VIP_Fine/")
        --> ["video1", "video10", "video30"]
        """
        images_dir = os.path.join(base_path, "Images")
        videos = os.listdir(images_dir)
        videos = list(filter(lambda x: "video" in x, videos))
        videos = sorted(videos)
        return videos


class VIPInstanceCropped(VIP):
    # TODO: it looks like instance annotations are not consistent within videos. This is a problem for the stochastic pair case
    # for example video 17, frame 826 -> frame 851. One instance is leaving the frame here
    # but the annotation marks the instance remaining in the frame as instance 1 in 851 and instance 2 in 826
    # one would have to use reid manually on the instances to get the assignment within the video
    def __init__(self, config):
        """
        Examples
        --------

            # mask out background - crop tight around instance
            d = VIPInstanceCropped({"masked": True})
            example = d.get_example(0)
            image = example["image"]
            plt.imshow((np.squeeze(image) + 1.0) / 2)
            plt.savefig("e1.png")

            # keep background - crop tight around instance
            d = VIPInstanceCropped({"masked": False})
            example = d.get_example(0)
            image = example["image"]
            plt.imshow((np.squeeze(image) + 1.0) / 2)
            plt.savefig("e2.png")

        :param config:
        """
        super(VIPInstanceCropped, self).__init__(config)
        self._masked = config.get("masked")

    def get_example(self, i):
        example = super(VIPInstanceCropped, self).get_example(i)
        instance_id = example["id_within_video"]
        image_path = example["file_path_"]
        instance_mask_path = example["human_id_label_path_"]
        image = edu.load_image(image_path)
        instance_mask = cv2.imread(instance_mask_path, -1)
        instance_mask = instance_id == instance_mask

        x, y, w, h = cv2.boundingRect(instance_mask.astype(np.uint8))
        image_crop = image[y : (y + h), x : (x + h), :]
        instance_mask_crop = instance_mask[y : (y + h), x : (x + h)]

        if self._masked:
            image = image_crop * np.expand_dims(instance_mask_crop, -1)
        else:
            image = image_crop

        image = edu.resize_float32(image, self.size)
        example["image"] = image
        return example


class VIPInstanceCroppedStochasticPair(VIPInstanceCropped, PRNGMixin):
    # TODO: it looks like instance annotations are not consistent within videos. This is a problem for the stochastic pair case
    # for example video 17, frame 826 -> frame 851. One instance is leaving the frame here
    # but the annotation marks the instance remaining in the frame as instance 1 in 851 and instance 2 in 826
    # one would have to use reid manually on the instances to get the assignment within the video
    def __init__(self, config):
        super(VIPInstanceCroppedStochasticPair, self).__init__(config)
        self.avoid_identity = config.get("data_avoid_identity", True)
        self.flip = config.get("data_flip", False)

        self.labels = edu.add_choices(self.labels, character_id_key="id")
        self.labels = pd.DataFrame(self.labels)

    def get_example(self, i):
        choices = self.labels["choices"][i]
        if self.avoid_identity and len(choices) > 1:
            choices = [c for c in choices if c != i]
        j = self.prng.choice(choices)

        e0 = super(VIPInstanceCroppedStochasticPair, self).get_example(i)
        e1 = super(VIPInstanceCroppedStochasticPair, self).get_example(j)

        return {"view0": e0["image"], "view1": e1["image"]}


if __name__ == "__main__":
    from pylab import *
    import cv2

    # d = VIP()
    # example = d.get_example(0)
    # image = cv2.imread(example["file_path_"])
    # plt.imshow(image)
    # plt.savefig("e0.png")
    #
    # d = VIPInstanceCropped({"masked": True})
    # example = d.get_example(0)
    # image = example["image"]
    # plt.imshow((np.squeeze(image) + 1.0) / 2)
    # plt.savefig("e1.png")
    #
    # d = VIPInstanceCropped({"masked": False})
    # example = d.get_example(0)
    # image = example["image"]
    # plt.imshow((np.squeeze(image) + 1.0) / 2)
    # plt.savefig("e2.png")

    d = VIPInstanceCroppedStochasticPair({"masked": True})
    example = d.get_example(2005)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow((example["view0"] + 1.0) / 2)
    axes[1].imshow((example["view1"] + 1.0) / 2)
    plt.savefig("e3.png")
