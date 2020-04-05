import os

import numpy as np
from mayavi import mlab

from waymo_extractor.protos.annotation_pb2 import Annotation


class Viewer3D:
    def __init__(self, args):
        self._base_dir = args.source
        self._label_dir = os.path.join(self._base_dir, "label")
        self._laser_dir = os.path.join(self._base_dir, "laser")
        self._laser_r_dir = os.path.join(self._base_dir, "laser_r")
        self._plane_dir = os.path.join(self._base_dir, "plane")
        self._files = [
            _.replace(".bin", "") for _ in os.listdir(self._laser_dir) if _.endswith("bin")
        ]
        self._current = 0
        self._reduce = args.reduce

    def _reduce(self, pcl: np.ndarray, plane: np.ndarray, threshold: float) -> np.ndarray:
        z_min = np.percentile(pcl[:, -1], 10)
        z_max = np.percentile(pcl[:, -1], 90)
        z_mask = np.logical_and(pcl[:, -1] > z_min, pcl[:, -1] < z_max)
        pcla = np.ones((len(pcl), 4), dtype=np.float32)
        pcla[:, :3] = pcl
        plane_mask = np.abs(pcla @ plane) > threshold
        mask = np.logical_and(z_mask, plane_mask)
        return mask

    def _plot_point_cloud(self, pcl: np.ndarray, pcl_r: np.ndarray, labels) -> None:
        radius = 0.05
        color = (1, 0.83, 0)
        mlab.points3d(
            pcl[:, 0],
            pcl[:, 1],
            pcl[:, 2],
            # (pcl_r[:, 0] % 85.0) / 85.0,
            pcl[:, 2],
            mode="point",
            colormap="jet",
            scale_factor=100,
            line_width=10,
            figure=mlab.figure(bgcolor=(0, 0, 0), size=(1920, 1080)),
        )

        for label in labels:
            box = label.box
            cx = box.center_x
            cy = box.center_y
            cz = box.center_z
            l = box.length  # noqa E741
            w = box.width
            h = box.height
            ry = box.heading
            x_corners = [l, l, -l, -l, l, l, -l, -l]
            y_corners = [w, -w, -w, w, w, -w, -w, w]
            z_corners = [h, h, h, h, -h, -h, -h, -h]
            R = np.array([[np.cos(ry), -np.sin(ry), 0], [np.sin(ry), np.cos(ry), 0], [0, 0, 1]])
            corners3d = np.vstack([x_corners, y_corners, z_corners]) / 2.0
            corners3d = (R @ corners3d).T + np.array([cx, cy, cz])
            x_corners, y_corners, z_corners = corners3d[:, 0], corners3d[:, 1], corners3d[:, 2]
            mlab.plot3d(
                [x_corners[0], x_corners[1]],
                [y_corners[0], y_corners[1]],
                [z_corners[0], z_corners[1]],
                color=color,
                tube_radius=radius,
            )
            mlab.plot3d(
                [x_corners[1], x_corners[2]],
                [y_corners[1], y_corners[2]],
                [z_corners[1], z_corners[2]],
                color=color,
                tube_radius=radius,
            )
            mlab.plot3d(
                [x_corners[2], x_corners[3]],
                [y_corners[2], y_corners[3]],
                [z_corners[2], z_corners[3]],
                color=color,
                tube_radius=radius,
            )
            mlab.plot3d(
                [x_corners[0], x_corners[3]],
                [y_corners[0], y_corners[3]],
                [z_corners[0], z_corners[3]],
                color=color,
                tube_radius=radius,
            )
            mlab.plot3d(
                [x_corners[4], x_corners[5]],
                [y_corners[4], y_corners[5]],
                [z_corners[4], z_corners[5]],
                color=color,
                tube_radius=radius,
            )
            mlab.plot3d(
                [x_corners[5], x_corners[6]],
                [y_corners[5], y_corners[6]],
                [z_corners[5], z_corners[6]],
                color=color,
                tube_radius=radius,
            )
            mlab.plot3d(
                [x_corners[6], x_corners[7]],
                [y_corners[6], y_corners[7]],
                [z_corners[6], z_corners[7]],
                color=color,
                tube_radius=radius,
            )
            mlab.plot3d(
                [x_corners[4], x_corners[7]],
                [y_corners[4], y_corners[7]],
                [z_corners[4], z_corners[7]],
                color=color,
                tube_radius=radius,
            )
            mlab.plot3d(
                [x_corners[0], x_corners[4]],
                [y_corners[0], y_corners[4]],
                [z_corners[0], z_corners[4]],
                color=color,
                tube_radius=radius,
            )
            mlab.plot3d(
                [x_corners[1], x_corners[5]],
                [y_corners[1], y_corners[5]],
                [z_corners[1], z_corners[5]],
                color=color,
                tube_radius=radius,
            )
            mlab.plot3d(
                [x_corners[2], x_corners[6]],
                [y_corners[2], y_corners[6]],
                [z_corners[2], z_corners[6]],
                color=color,
                tube_radius=radius,
            )
            mlab.plot3d(
                [x_corners[3], x_corners[7]],
                [y_corners[3], y_corners[7]],
                [z_corners[3], z_corners[7]],
                color=color,
                tube_radius=radius,
            )
        mlab.show()

    def display(self, step: int = 1) -> None:
        total = len(self._files)

        while True:
            filename = self._files[self._current % total]
            laser_path = os.path.join(self._laser_dir, "{}.bin".format(filename))
            laser_r_path = os.path.join(self._laser_r_dir, "{}.bin".format(filename))
            label_path = os.path.join(self._label_dir, "{}.pb".format(filename))

            pcl = np.fromfile(laser_path, dtype=np.float32).reshape(-1, 3)
            pcl_r = np.fromfile(laser_r_path, dtype=np.float32).reshape(-1, 1)

            if self._reduce:
                plane_path = os.path.join(self._plane_dir, "{}.txt".format(filename))
                if not os.path.isfile(plane_path):
                    continue
                plane = np.loadtxt(plane_path, dtype=np.float32)
                mask = self._reduce(pcl, plane, 0.1)
                pcl = pcl[mask]
                pcl_r = pcl_r[mask]

            anno = Annotation()
            anno.ParseFromString(open(label_path, "rb").read())

            labels = anno.laser_labels

            self._plot_point_cloud(pcl, pcl_r, labels)

            self._current += step
