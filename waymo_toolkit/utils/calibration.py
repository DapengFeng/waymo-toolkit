import numpy as np


def get_image_transform(camera_calibration):
    """ For a given camera calibration, compute the transformation matrix
        from the vehicle reference frame to the image space.
    """

    extrinsic = np.array(camera_calibration.extrinsic.transform).reshape(4, 4)
    intrinsic = camera_calibration.intrinsic

    camera_model = np.array(
        [[intrinsic[0], 0, intrinsic[2], 0], [0, intrinsic[1], intrinsic[3], 0], [0, 0, 1, 0]]
    )

    # Swap the axes around
    axes_transformation = np.array([[0, -1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0], [0, 0, 0, 1]])

    # Compute the projection matrix from the vehicle space to image space.
    vehicle_to_image = np.matmul(
        camera_model, np.matmul(axes_transformation, np.linalg.inv(extrinsic))
    )
    return vehicle_to_image
