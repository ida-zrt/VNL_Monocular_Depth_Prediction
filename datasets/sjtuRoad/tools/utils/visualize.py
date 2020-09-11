import mayavi.mlab as mlab
import numpy as np
import math


def showPoints(points, fig):
    xs = points[:, 0]
    ys = points[:, 1]
    zs = points[:, 2]
    col = zs
    mlab.points3d(xs, ys, zs, col, mode="point", figure=fig)


def createRotation(angles1):
    theta = np.zeros((3, 1), dtype=np.float64)
    theta[0] = angles1[0] * 3.141592653589793 / 180.0
    theta[1] = angles1[1] * 3.141592653589793 / 180.0
    theta[2] = angles1[2] * 3.141592653589793 / 180.0
    R_x = np.array([[1, 0, 0], [0, math.cos(theta[0]), -math.sin(theta[0])],
                    [0, math.sin(theta[0]),
                     math.cos(theta[0])]])
    R_y = np.array([[math.cos(theta[1]), 0,
                     math.sin(theta[1])], [0, 1, 0],
                    [-math.sin(theta[1]), 0,
                     math.cos(theta[1])]])
    R_z = np.array([[math.cos(theta[2]), -math.sin(theta[2]), 0],
                    [math.sin(theta[2]),
                     math.cos(theta[2]), 0], [0, 0, 1]])
    R = np.dot(R_z, np.dot(R_y, R_x))
    return R


def plot3Dbox(corners, fig):
    for i in range(corners.shape[0]):
        corner = corners[i]
        idx = np.array([0, 1, 2, 3, 0, 4, 5, 6, 7, 4, 5, 1, 2, 6, 7, 3])
        x = corner[idx, 0]
        y = corner[idx, 1]
        z = corner[idx, 2]
        mlab.plot3d(x,
                    y,
                    z,
                    color=(1, 0, 0),
                    colormap='Spectral',
                    representation='wireframe',
                    line_width=2,
                    figure=fig)


def boxes3d_to_corners3d(boxes3d, rotate=True):
    """
    :param boxes3d: (N, 7) [x, y, z, h, w, l, ry]
    :param rotate:
    :return: corners3d: (N, 8, 3)
    """
    boxes_num = boxes3d.shape[0]
    h, w, l = boxes3d[:, 3], boxes3d[:, 4], boxes3d[:, 5]
    x_corners = np.array(
        [l / 2., l / 2., -l / 2., -l / 2., l / 2., l / 2., -l / 2., -l / 2.],
        dtype=np.float32).T  # (N, 8)
    z_corners = np.array(
        [w / 2., -w / 2., -w / 2., w / 2., w / 2., -w / 2., -w / 2., w / 2.],
        dtype=np.float32).T  # (N, 8)

    y_corners = np.zeros((boxes_num, 8), dtype=np.float32)
    y_corners[:, 4:8] = -h.reshape(boxes_num, 1).repeat(4, axis=1)  # (N, 8)

    if rotate:
        ry = boxes3d[:, 6]
        zeros, ones = np.zeros(ry.size,
                               dtype=np.float32), np.ones(ry.size,
                                                          dtype=np.float32)
        rot_list = np.array([[np.cos(ry), zeros, -np.sin(ry)],
                             [zeros, ones, zeros],
                             [np.sin(ry), zeros, np.cos(ry)]])  # (3, 3, N)
        R_list = np.transpose(rot_list, (2, 0, 1))  # (N, 3, 3)

        temp_corners = np.concatenate(
            (x_corners.reshape(-1, 8, 1), y_corners.reshape(
                -1, 8, 1), z_corners.reshape(-1, 8, 1)),
            axis=2)  # (N, 8, 3)
        rotated_corners = np.matmul(temp_corners, R_list)  # (N, 8, 3)
        x_corners, y_corners, z_corners = rotated_corners[:, :,
                                                          0], rotated_corners[:, :,
                                                                              1], rotated_corners[:, :,
                                                                                                  2]

    x_loc, y_loc, z_loc = boxes3d[:, 0], boxes3d[:, 1], boxes3d[:, 2]

    x = x_loc.reshape(-1, 1) + x_corners.reshape(-1, 8)
    y = y_loc.reshape(-1, 1) + y_corners.reshape(-1, 8)
    z = z_loc.reshape(-1, 1) + z_corners.reshape(-1, 8)

    corners = np.concatenate(
        (x.reshape(-1, 8, 1), y.reshape(-1, 8, 1), z.reshape(-1, 8, 1)),
        axis=2)

    return corners.astype(np.float32)
