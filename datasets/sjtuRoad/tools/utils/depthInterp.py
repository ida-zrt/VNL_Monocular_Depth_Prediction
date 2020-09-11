from __future__ import print_function
from scipy.interpolate import griddata
import cv2 as cv
import numpy as np


def linearFit(X, Y, ValidDepth, xs, ys, cell):
    # model: z = a + b*u + c*v
    if ValidDepth.shape[0] <= 2:
        return None
    # solve params in Least Square Manner
    mat = np.concatenate([np.ones_like(X), X, Y], axis=1)
    if abs(np.linalg.det(mat)) < 1e-10:
        raise Exception('Singular mat')
    q, r = np.linalg.qr(mat)
    b = np.dot(q.T, ValidDepth)
    params = np.linalg.solve(r, b).squeeze()

    # calculate missing depths
    depths = params[0] + params[1] * xs + params[2] * ys
    if any(depths > 65536):
        raise Exception('Overflow Err')
    return depths


def newFit(X, Y, ValidDepth, xs, ys):
    if ValidDepth.shape[0] <= 3:
        return None
    # model: z = a + b*u + c*v + d*u*v
    # solve params in Least Square Manner
    mat = np.concatenate([np.ones_like(X), X, Y, X * Y], axis=1)
    if abs(np.linalg.det(mat)) < 1e-10:
        raise Exception('Singular mat')
    q, r = np.linalg.qr(mat)
    b = np.dot(q.T, ValidDepth)
    params = np.linalg.solve(r, b).squeeze()

    # calculate missing depths
    depths = params[0] + params[1] * xs + params[2] * ys + params[3] * xs * ys
    if any(depths > 65536):
        raise Exception('Overflow Err')
    return depths


def _cellInterp(cell, mode='bilinear'):
    ValidDepth = np.expand_dims(cell[cell > 0].reshape(-1), 1)
    # return if all is valid or invalid
    if ValidDepth.shape[0] == 0:
        return cell
    if ValidDepth.shape[0] == cell.size:
        return cell

    # fit for missing values in the kernel
    xs = np.arange(cell.shape[1])
    ys = np.arange(cell.shape[0])
    X, Y = np.meshgrid(xs, ys)
    X = X.reshape(-1)
    Y = Y.reshape(-1)

    # find valid axis to solve fitting model
    isValid = (cell > 0).reshape(-1)
    X = np.expand_dims(X[isValid], 1)
    Y = np.expand_dims(Y[isValid], 1)

    xs2, ys2 = np.meshgrid(xs, ys)
    try:
        if mode.lower() == 'linear':
            depths = linearFit(X, Y, ValidDepth, xs2, ys2, cell)
        elif mode.lower() == 'bilinear':
            try:
                depths = newFit(X, Y, ValidDepth, xs2, ys2, cell)
            except:
                depths = linearFit(X, Y, ValidDepth, xs2, ys2, cell)
        if depths is None:
            raise Exception('Not enough data')
    except Exception as err:
        # print(err)
        depths = cell
        depths[depths == 0] = np.mean(ValidDepth)
        return depths

    return depths.reshape(cell.shape)


def fitDepthImg(sparseDepth, ksize, mode='bilinear'):
    if ksize == 0:
        return sparseDepth
    if len(ksize) == 1:
        ksize = (ksize, ksize)
    if ksize[0] == 0 or ksize[1] == 0:
        return sparseDepth
    d1 = 0
    d2 = 0
    if sparseDepth.shape[0] % ksize[0]:
        d1 = ksize[0] - sparseDepth.shape[0] % ksize[0]
    if sparseDepth.shape[1] % ksize[1]:
        d2 = ksize[1] - sparseDepth.shape[1] % ksize[1]

    sDepth = np.pad(sparseDepth, ((0, d1), (0, d2)), 'constant').astype(float)
    yn = int(sDepth.shape[0] / ksize[0])
    xn = int(sDepth.shape[1] / ksize[1])
    for y in range(yn):
        for x in range(xn):
            cell = sDepth[y * ksize[0]:(y + 1) * ksize[0],
                          x * ksize[1]:(x + 1) * ksize[1]]
            cell = _cellInterp(cell, mode)
            sDepth[y * ksize[0]:(y + 1) * ksize[0],
                   x * ksize[1]:(x + 1) * ksize[1]] = cell

    sDepth = sDepth[0:sparseDepth.shape[0], 0:sparseDepth.shape[1]]
    return sDepth


def interpDepthImg(depthImg, ksize=(3, 5), method='linear', fill_value=0):
    depthImg = fitDepthImg(depthImg, ksize)
    points = np.array((depthImg != 0).nonzero()).T
    validDepth = depthImg[depthImg != 0]
    xs = np.arange(depthImg.shape[0])
    ys = np.arange(depthImg.shape[1])
    X, Y = np.meshgrid(xs, ys)
    X = X.reshape(-1)
    Y = Y.reshape(-1)
    depth = griddata(points, validDepth, (X, Y), method, fill_value)
    depth = depth.reshape(depthImg.shape[::-1]).T
    depth[depth < 0] = 0
    return depth


if __name__ == "__main__":
    sparseDepth = cv.imread('../data_2020_07_31_10_46_48/depthRaw/depthRaw0002.png',
                            -1)
    depth = fitDepthImg(sparseDepth, (8, 8))
    cv.imwrite('./depthFitting.png', depth.astype(np.uint16))
    depth2 = interpDepthImg(sparseDepth, 0)
    cv.imwrite('./pureInterp.png', depth2.astype(np.uint16))
    depth3 = interpDepthImg(depth, 0)
    cv.imwrite('./interpFit.png', depth3.astype(np.uint16))
