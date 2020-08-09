from __future__ import print_function
import cv2 as cv
import numpy as np


def getDepthFromCloud(cloudPoints,
                      rvec,
                      tvec,
                      cameraMat,
                      distCoeff,
                      imgSize,
                      maxDepth=280):
    imgPoints, _ = cv.projectPoints(cloudPoints, rvec, tvec, cameraMat,
                                    distCoeff)
    imgPoints = np.squeeze(imgPoints)
    x_inrange = np.logical_and(imgPoints[:, 0] < imgSize[0],
                               imgPoints[:, 0] >= 0)
    y_inrange = np.logical_and(imgPoints[:, 1] < imgSize[1],
                               imgPoints[:, 1] >= 0)
    inRange = np.logical_and(x_inrange, y_inrange)
    imgPoints = imgPoints[inRange, :].astype(int)
    rotMat, _ = cv.Rodrigues(rvec)

    depth = np.matmul(rotMat, cloudPoints.T)[2]
    depth = depth[inRange]

    depthImg = np.zeros(imgSize)
    depthImg[imgPoints[:, 0], imgPoints[:, 1]] = depth
    print('depth mean is {} m'.format(np.mean(depth)))
    print('depth var is {} m2'.format(np.var(depth)))
    print('depth max is {} m'.format(np.max(depth)))
    print('depth min is {} m'.format(np.min(depth[depth != 0])))
    crit = 2 * np.sqrt(np.var(depth)) + np.mean(depth)
    percent = np.sum(depth < crit) / np.sum(np.ones_like(depth))
    print('{:.2f} %% depth is below {} m'.format(percent * 100, crit))
    print('')
    depthImg = (np.transpose(depthImg) * 65536 / maxDepth).astype(np.uint16)
    return depthImg


def generateColoredDepthImg(depthImg):
    depthHue = (depthImg.astype(float) / np.max(depthImg) * 180)
    v = np.expand_dims(np.ones((depthHue.shape)), 2) * \
        np.expand_dims(depthHue.astype(np.uint8), 2)
    v = v / np.max(v) * 50 + 180
    v[v == 180] = 0
    v[depthImg > 23400] = 180
    s = np.expand_dims(np.ones((depthHue.shape)), 2) * 230
    s[depthImg > 23400] = 0
    hsv = np.concatenate([np.expand_dims(depthHue.astype(np.uint8), 2), s, v],
                         2).astype(np.uint8)

    coloredDepthImg = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
    return coloredDepthImg
