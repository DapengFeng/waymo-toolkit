import numpy as np


def get_box_transformation_matrix(box):
    """Create a transformation matrix for a given label box pose."""

    tx, ty, tz = box.center_x, box.center_y, box.center_z
    c = np.cos(box.heading)
    s = np.sin(box.heading)

    sl, sh, sw = box.length, box.height, box.width

    return np.array(
        [[sl * c, -sw * s, 0, tx], [sl * s, sw * c, 0, ty], [0, 0, sh, tz], [0, 0, 0, 1]]
    )


def get_3d_box_projected_corners(vehicle_to_image, label):
    """Get the 2D coordinates of the 8 corners of a label's 3D bounding box.
    vehicle_to_image: Transformation matrix from the vehicle frame to the image frame.
    label: The object label
    """

    box = label.box

    # Get the vehicle pose
    box_to_vehicle = get_box_transformation_matrix(box)

    # Calculate the projection from the box space to the image space.
    box_to_image = np.matmul(vehicle_to_image, box_to_vehicle)

    # Loop through the 8 corners constituting the 3D box
    # and project them onto the image
    vertices = np.empty([2, 2, 2, 2])
    for k in [0, 1]:
        for l in [0, 1]:
            for m in [0, 1]:
                # 3D point in the box space
                v = np.array([(k - 0.5), (l - 0.5), (m - 0.5), 1.0])

                # Project the point onto the image
                v = np.matmul(box_to_image, v)

                # If any of the corner is behind the camera, ignore this object.
                if v[2] < 0:
                    return None

                vertices[k, l, m, :] = [v[0] / v[2], v[1] / v[2]]

    vertices = vertices.astype(np.int32)

    return vertices
