import argparse
import multiprocessing as mp
import os

import numpy as np

from waymo_toolkit.utils.ransac import ransac


def run(filename, laser_dir, save_dir, iterations, threshold, stop, seed):
    print(filename)
    laser_path = os.path.join(laser_dir, "{}.bin".format(filename))
    save_path = os.path.join(save_dir, "{}.txt".format(filename))
    laser = np.fromfile(laser_path, dtype=np.float32).reshape(-1, 3)
    m = ransac(laser, int(0.2 * len(laser)), iterations, threshold, stop, seed)
    np.savetxt(save_path, m)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source", required=True, help="provide source path to visualize", type=str
    )
    parser.add_argument("--iterations", default=100, help="the max number of interations", type=int)
    parser.add_argument("--threshold", default=0.1, help="threshold", type=float)
    parser.add_argument(
        "--stop", default=False, help="stop at the inliers surpass the upper", type=bool
    )
    parser.add_argument("--seed", default=20200319, help="random seed for ransac", type=int)
    args = parser.parse_args()

    base_dir = args.source
    laser_dir = os.path.join(base_dir, "laser")
    save_dir = os.path.join(base_dir, "plane")
    os.makedirs(save_dir, exist_ok=True)
    files = [int(_.replace(".bin", "")) for _ in os.listdir(laser_dir) if _.endswith("bin")]
    total = len(files)

    pool = mp.Pool(mp.cpu_count())
    pool.starmap_async(
        run,
        zip(
            files,
            [laser_dir] * total,
            [save_dir] * total,
            [args.iterations] * total,
            [args.threshold] * total,
            [args.stop] * total,
            [args.seed] * total,
        ),
    )
    pool.close()
    pool.join()


if __name__ == "__main__":
    main()
