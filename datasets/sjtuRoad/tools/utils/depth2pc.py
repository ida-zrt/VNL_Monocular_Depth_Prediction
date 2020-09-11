# load depth pics or pcd files
import torch
import numpy as np
import cv2 as cv
import glob
# import pcl


def depth2points(depthImg, fx, fy, cx, cy, maxDepth):
    z = depthImg / 65536 * maxDepth
    xs = np.arange(z.shape[1])
    ys = np.arange(z.shape[0])
    xs = (xs - cx) / fx
    ys = (ys - cy) / fy
    X, Y = np.meshgrid(xs, ys)
    points = np.transpose((np.array([X * z, Y * z, z])), (1, 2, 0))
    return points[depthImg != -1]


def depth2points2(depthImg, calibfile, maxDepth):
    fs = cv.FileStorage(calibfile, cv.FILE_STORAGE_READ)
    cameraMat = fs.getNode('CameraMat').mat()
    # distCoeff = fs.getNode('DistCoeff').mat()
    fs.release()
    fx = cameraMat[0, 0]
    fy = cameraMat[1, 1]
    cx = cameraMat[0, 2]
    cy = cameraMat[1, 2]

    points = depth2points(depthImg, fx, fy, cx, cy, maxDepth).reshape((-1, 3))

    return points
