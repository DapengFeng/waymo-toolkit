import argparse
import os

import numpy as np
import tensorflow as tf

from waymo_toolkit.extractor import ImageExtractor, LabelExtractor, LaserExtractor
from waymo_toolkit.utils.logger import setup_logger

logger = setup_logger("extractor")


def extract(source_dir: str, save_dir: str, args):
    files = os.listdir(source_dir)
    files = [os.path.join(source_dir, _) for _ in files if _.endswith("tfrecord")]
    assert len(files) > 0

    dataset = tf.data.TFRecordDataset(files)

    if args.image:
        imgext = ImageExtractor(dataset, save_dir)
        imgext.extract()

    if args.label:
        labext = LabelExtractor(dataset, save_dir)
        labext.extract()

    if args.laser:
        lasext = LaserExtractor(dataset, save_dir)
        lasext.extract()

    if args.subset:
        seed = args.seed
        percentage = args.percentage

        files = sorted(
            [int(_.replace(".pb", "")) for _ in os.listdir(os.path.join(save_dir, "label"))]
        )
        files = np.array(files)
        total = len(files)
        logger.info(
            "{} frames in {} totally".format(total, save_dir)
        )  # training: 158081  # validation: 39987

        np.random.seed(seed)
        index = np.random.choice(total, int(percentage * total), replace=False)
        selected = files[index]
        np.savetxt(os.path.join(save_dir, "split.txt"), selected)


def main():
    parser = argparse.ArgumentParser(description="Extracting the elements from Waymo")
    parser.add_argument("--source", required=True, help="provide source path to waymo", type=str)
    parser.add_argument("--dest", required=True, help="provide destination path", type=str)
    parser.add_argument(
        "--type",
        default="train",
        help="type of the extrated data, in ['train', 'val', 'test', 'all']",
        type=str,
    )
    parser.add_argument("--image", action="store_true", help="whether to extract images")
    parser.add_argument("--label", action="store_true", help="whether to extract labels")
    parser.add_argument("--laser", action="store_true", help="whether to extract lasers")
    parser.add_argument("--subset", action="store_true", help="whether to extract the subset")
    parser.add_argument(
        "--seed", default=20200319, help="random seed for select the subset", type=int
    )
    parser.add_argument("--percentage", default=0.1, help="the percentage of subset", type=float)

    args = parser.parse_args()

    source_dir = args.source
    save_dir = args.dest
    assert args.type in ["train", "val", "test", "all"]

    if args.type == "all":
        for fold in ["training_seg", "validation_seg", "testing_seg"]:
            source_folder = os.path.join(source_dir, fold)
            save_folder = os.path.join(save_dir, fold.replace("_seg", ""))
            if fold == "testing_seg":
                args.subset = False
            extract(source_folder, save_folder, args)
    elif args.type == "train":
        source_dir = os.path.join(source_dir, "training_seg")
        save_dir = os.path.join(save_dir, "training")
    elif args.type == "val":
        source_dir = os.path.join(source_dir, "validation_seg")
        save_dir = os.path.join(save_dir, "validation")
    elif args.type == "test":
        source_dir = os.path.join(source_dir, "testing_seg")
        save_dir = os.path.join(save_dir, "testing")
    extract(source_dir, save_dir, args)


if __name__ == "__main__":
    main()
