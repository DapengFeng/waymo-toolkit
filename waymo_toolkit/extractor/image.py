import os

import cv2
import numpy as np
import tensorflow as tf

from waymo_toolkit.protos import dataset_pb2 as open_dataset
from waymo_toolkit.utils.logger import setup_logger

from .extractor import Extractor

logger = setup_logger("extractor")


class ImageExtractor(Extractor):
    def __init__(self, dataset: tf.data.TFRecordDataset, save_dir: str):
        super(ImageExtractor, self).__init__(dataset, save_dir)

    def _save(self, path: str, data: np.ndarray) -> None:
        self._cnt += 1
        if self._cnt % 1000 == 1:
            logger.info("{:08d} : {}".format(self._cnt, path))
        cv2.imwrite(path, data)

    def extract(self) -> None:
        for data in self._dataset:
            frame = open_dataset.Frame()
            frame.ParseFromString(bytearray(data.numpy()))
            for i, im in enumerate(frame.images):
                ima = tf.image.decode_jpeg(im.image).numpy()[:, :, ::-1]
                base_name = "{}.jpg".format(frame.timestamp_micros)
                base_dir = os.path.join(self._save_dir, "image_{}".format(i))
                os.makedirs(base_dir, exist_ok=True)
                filename = os.path.join(base_dir, base_name)
                self._save(filename, ima)
        logger.info("Finish extracting image")
