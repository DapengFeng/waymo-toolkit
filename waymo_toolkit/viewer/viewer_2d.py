import os
from typing import Tuple

import cv2
import matplotlib.cm
import numpy as np

from waymo_toolkit.protos.annotation_pb2 import Annotation
from waymo_toolkit.utils.box_utils import get_3d_box_projected_corners
from waymo_toolkit.utils.calibration import get_image_transform

cmap = matplotlib.cm.get_cmap("jet")


class Viewer2D:
    def __init__(self, args):
        self._base_dir = args.source
        self._draw_2d = args.box2d
        self._draw_3d = args.box3d
        self._project = args.project
        self._label_dir = os.path.join(self._base_dir, "label")
        self._laser_dir = os.path.join(self._base_dir, "laser")
        self._laser_r_dir = os.path.join(self._base_dir, "laser_r")
        self._files = [
            _.replace(".bin", "") for _ in os.listdir(self._laser_dir) if _.endswith("bin")
        ]
        self._current = 0

    def _draw_2d_box(
        self, img: np.ndarray, label, colour: Tuple[int, int, int] = (0x00, 0xD3, 0xFF)
    ) -> None:
        """Draw a 2D bounding from a given 2D label on a given "img".
        """

        box = label.box

        # Extract the 2D coordinates
        # It seems that "length" is the actual width
        # and "width" is the actual height of the bounding box. Most peculiar.
        x1 = int(box.center_x - box.length / 2)
        x2 = int(box.center_x + box.length / 2)
        y1 = int(box.center_y - box.width / 2)
        y2 = int(box.center_y + box.width / 2)

        # Draw the rectangle
        cv2.rectangle(img, (x1, y1), (x2, y2), colour, thickness=2)

    def _draw_3d_box(
        self,
        img: np.ndarray,
        vehicle_to_image: np.ndarray,
        label,
        color: Tuple[int, int, int] = (0xAF, 0x14, 0x39),
    ) -> None:
        vertices = get_3d_box_projected_corners(vehicle_to_image, label)
        if vertices is None:
            # The box is not visible in this image
            return

        # Draw the edges of the 3D bounding box
        for k in [0, 1]:
            for l in [0, 1]:
                for idx1, idx2 in [
                    ((0, k, l), (1, k, l)),
                    ((k, 0, l), (k, 1, l)),
                    ((k, l, 0), (k, l, 1)),
                ]:
                    cv2.line(img, tuple(vertices[idx1]), tuple(vertices[idx2]), color, thickness=2)
        # Draw a cross on the front face to identify front & back.
        for idx1, idx2 in [((1, 0, 0), (1, 1, 1)), ((1, 1, 0), (1, 0, 1))]:
            cv2.line(img, tuple(vertices[idx1]), tuple(vertices[idx2]), color, thickness=2)

    def _project_laser_on_image(
        self, img: np.ndarray, pcl: np.ndarray, pcl_r: np.ndarray, vehicle_to_image: np.ndarray,
    ) -> None:
        # Convert the pointcloud to homogeneous coordinates.
        pcl1 = np.concatenate((pcl, np.ones_like(pcl[:, 0:1])), axis=1)

        # Transform the point cloud to image space.
        pcl_cp = np.einsum("ij,bj->bi", vehicle_to_image, pcl1)

        # Filter LIDAR points which are behind the camera.
        mask = pcl_cp[:, 2] > 0
        pcl_cp = pcl_cp[mask]
        pcl_r = pcl_r[mask]

        # Project the point cloud onto the image.
        pcl_cp = pcl_cp[:, :2] / pcl_cp[:, 2:3]

        # Filter points which are outside the image.
        mask = np.logical_and(
            np.logical_and(pcl_cp[:, 0] > 0, pcl_cp[:, 0] < img.shape[1]),
            np.logical_and(pcl_cp[:, 1] > 0, pcl_cp[:, 1] < img.shape[1]),
        )

        pcl_cp = pcl_cp[mask]
        pcl_r = pcl_r[mask]

        # Colour code the points based on distance.
        coloured_intensity = 255 * cmap((pcl_r[:, 0] % 85.0) / 85.0)

        # Draw a circle for each point.
        for i in range(pcl_cp.shape[0]):
            cv2.circle(img, (int(pcl_cp[i, 0]), int(pcl_cp[i, 1])), 1, coloured_intensity[i])

    def display(self, camera: int = 0, step: int = 1) -> None:
        self._image_dir = os.path.join(self._base_dir, "image_{}".format(camera))
        total = len(self._files)

        while True:
            filename = self._files[self._current % total]
            image_path = os.path.join(self._image_dir, "{}.jpg".format(filename))
            label_path = os.path.join(self._label_dir, "{}.pb".format(filename))
            laser_path = os.path.join(self._laser_dir, "{}.bin".format(filename))
            laser_r_path = os.path.join(self._laser_r_dir, "{}.bin".format(filename))

            img = cv2.imread(image_path)
            pcl = np.fromfile(laser_path, dtype=np.float32).reshape(-1, 3)
            pcl_r = np.fromfile(laser_r_path, dtype=np.float32).reshape(-1, 1)

            anno = Annotation()
            anno.ParseFromString(open(label_path, "rb").read())

            camera_calibration = anno.context.camera_calibrations[camera]
            vehicle_to_image = get_image_transform(camera_calibration)

            if self._project:
                self._project_laser_on_image(img, pcl, pcl_r, vehicle_to_image)

            if self._draw_2d:
                labels = anno.camera_labels[camera].labels
                for label in labels:
                    self._draw_2d_box(img, label)

            if self._draw_3d:
                labels = anno.laser_labels
                for label in labels:
                    self._draw_3d_box(img, vehicle_to_image, label)

            cv2.imshow("Viewer2D", img)
            key = cv2.waitKey(-1)
            if key == ord("q"):
                break
            elif key == ord("s"):
                cv2.imsave("demo{}.jpg".format(anno.timestamp_micros), img)

            self._current += step
