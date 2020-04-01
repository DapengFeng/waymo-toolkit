import os

import tensorflow as tf

from waymo_extractor.protos import annotation_pb2 as annotation
from waymo_extractor.utils.logger import setup_logger

from .extractor import Extractor

logger = setup_logger("extractor")


class LabelExtractor(Extractor):
    def __init__(self, dataset: tf.data.TFRecordDataset, save_dir: str):
        super(LabelExtractor, self).__init__(dataset, save_dir)

    def _save(self, path: str, data: annotation.Annotation) -> None:
        msg = data.SerializeToString()
        self._cnt += 1
        if self._cnt % 1000 == 1:
            logger.info("{:08d} : {}".format(self._cnt, path))
        with open(path, "wb") as fout:
            fout.write(msg)
            fout.close()

    def extract(self) -> None:
        for data in self._dataset:
            anno = annotation.Annotation()
            anno.ParseFromString(bytearray(data.numpy()))
            anno.DiscardUnknownFields()
            base_name = "{}.pb".format(anno.timestamp_micros)
            base_dir = os.path.join(self._save_dir, "label")
            os.makedirs(base_dir, exist_ok=True)
            filename = os.path.join(base_dir, base_name)
            self._save(filename, anno)
        logger.info("Finish extracting label")
