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
    depthImg = (np.transpose(depthImg) * 65536 / maxDepth).astype(np.uint16)
    return depthImg


def generateColoredDepthImg(depthImg):
    depthHue = (depthImg.astype(float) / np.max(depthImg) * 180)
    v = np.expand_dims(np.ones((depthHue.shape)), 2) * \
        np.expand_dims(depthHue.astype(np.uint8), 2)
    v = v / np.max(v) * 50 + 180
    v[v == 180] = 0
    hsv = np.concatenate([
        np.expand_dims(depthHue.astype(np.uint8), 2),
        np.expand_dims(np.ones((depthHue.shape)), 2) * 230, v
    ], 2).astype(np.uint8)

    coloredDepthImg = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
    return coloredDepthImg
