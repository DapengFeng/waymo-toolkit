import os

import numpy as np
import tensorflow as tf

from waymo_extractor.protos import dataset_pb2 as open_dataset
from waymo_extractor.utils.frame_utils import convert_range_image_to_point_cloud
from waymo_extractor.utils.logger import setup_logger
from waymo_open_dataset.utils.frame_utils import parse_range_image_and_camera_projection

from .extractor import Extractor

logger = setup_logger("extractor")


class LaserExtractor(Extractor):
    def __init__(self, dataset: tf.data.TFRecordDataset, save_dir: str):
        super(LaserExtractor, self).__init__(dataset, save_dir)

    def _save(self, path: str, data: np.ndarray) -> None:
        self._cnt += 1
        if self._cnt % 1000 == 1:
            logger.info("{:08d} : {}".format(self._cnt, path))
        data.flatten().tofile(path)

    def extract(self) -> None:
        for data in self._dataset:
            frame = open_dataset.Frame()
            frame.ParseFromString(bytearray(data.numpy()))
            (
                range_images,
                camera_projections,
                range_image_top_pose,
            ) = parse_range_image_and_camera_projection(frame)
            points, r_points, i_points, e_points, cp_points = convert_range_image_to_point_cloud(
                frame, range_images, camera_projections, range_image_top_pose
            )

            base_name = "{}.bin".format(frame.timestamp_micros)

            # cartesian
            base_dir = os.path.join(self._save_dir, "laser")
            os.makedirs(base_dir, exist_ok=True)
            filename = os.path.join(base_dir, base_name)
            self._save(filename, points)

            # range
            base_dir = os.path.join(self._save_dir, "laser_r")
            os.makedirs(base_dir, exist_ok=True)
            filename = os.path.join(base_dir, base_name)
            self._save(filename, r_points)

            # intensity
            base_dir = os.path.join(self._save_dir, "laser_i")
            os.makedirs(base_dir, exist_ok=True)
            filename = os.path.join(base_dir, base_name)
            self._save(filename, i_points)

            # elongation
            base_dir = os.path.join(self._save_dir, "laser_e")
            os.makedirs(base_dir, exist_ok=True)
            filename = os.path.join(base_dir, base_name)
            self._save(filename, e_points)

            # cemera projection
            base_dir = os.path.join(self._save_dir, "laser_cp")
            os.makedirs(base_dir, exist_ok=True)
            filename = os.path.join(base_dir, base_name)
            self._save(filename, cp_points)
        logger.info("Finish extracting laser")
