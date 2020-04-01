from typing import Any

import tensorflow as tf


class Extractor:
    def __init__(self, dataset: tf.data.TFRecordDataset, save_dir: str):
        self._dataset = dataset
        self._save_dir = save_dir
        self._cnt = 0

    def _save(self, path: str, data: Any) -> None:
        raise NotImplementedError

    def extract(self) -> None:
        raise NotImplementedError
