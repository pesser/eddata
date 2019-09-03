import pandas as pd
import os
from pathlib import Path
import numpy as np
import eddata.utils as edu
import traceback

from eddata.utils import df_empty


class VIP(edu.DatasetMixin):
    def __init__(self, config=None):
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
        self._prepare()
        self._load()

    def _prepare(self):
        self.root = edu.get_root("vip_fine")
        self._label_path = Path(self.root).joinpath("eddata_labels.csv")
        if not edu.is_prepared(self.root):
            # dataframe layout:
            # global_id | id_within_video | videoXXX
            global_ids_csv = os.path.join(self.root, "global_ids.csv")
            if os.path.exists(global_ids_csv):
                df_global_ids = pd.read_csv(global_ids_csv)
            else:
                df_global_ids = self.make_global_human_ids()
                df_global_ids.to_csv(global_ids_csv, index=False)
            df_labels = df_empty(
                ["id", "relative_file_path_", "relative_human_id_label_path_"]
            )
            for _, (global_id, id_within_video, video) in df_global_ids.iterrows():
                new_data = {
                    "id": global_id,
                    "relative_file_path_": self.list_image_files(video),
                    "relative_human_id_label_path_": self.list_human_instance_segmentation_files(
                        video
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
        columns = ["global_id", "id_within_video", "video"]
        df = df_empty(columns, [np.int64, np.int64, str])
        videos = self.list_videos(base_path)
        failed_videos = []
        successfull_videos = []
        for video in videos:
            try:
                instance_id_annotation_file = self.get_instance_id_annotation_file(
                    base_path, video
                )
                human_ids_within_video = self.get_human_ids_within_video(
                    instance_id_annotation_file
                )
                new_global_human_ids = list(
                    map(lambda v_id: v_id + global_human_id, human_ids_within_video)
                )
                msg = "visiting {} - found ids {} - adding as {}".format(
                    video, human_ids_within_video, new_global_human_ids
                )
                print(msg)
                for new_global_human_id, new_id_within_video in zip(
                    new_global_human_ids, human_ids_within_video
                ):
                    new_data_row = {
                        "global_id": new_global_human_id,
                        "id_within_video": new_id_within_video,
                        "video": video,
                    }
                    df = df.append(new_data_row, ignore_index=True)
                global_human_id = max(new_global_human_ids)
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

    def get_instance_id_annotation_file(self, base_path, video):
        """get_instance_id_annotation_file("/mnt/hci_gpu/compvis_group/sabraun/LIP_VIP/VIP_Fine", "videos1"):
            --> /mnt/hci_gpu/compvis_group/sabraun/LIP_VIP/VIP_Fine/Annotations/Instance_ids/videos1/000000000001.txt"
        """
        abs_path = os.path.join(base_path, "Annotations", "Instance_ids", video)
        txt_files = os.listdir(abs_path)
        txt_files = list(filter(lambda x: os.path.splitext(x)[1] == ".txt", txt_files))
        first_file = sorted(txt_files)[0]
        return os.path.join(abs_path, first_file)

    def get_human_ids_within_video(self, instance_id_annotation_file):
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


if __name__ == "__main__":
    from matplotlib import pyplot as plt
    import cv2

    config = {""}
    d = VIP()
    example = d.get_example(0)
    image = cv2.imread(example["file_path_"])
    plt.imshow(image)
    plt.savefig("e0.png")
