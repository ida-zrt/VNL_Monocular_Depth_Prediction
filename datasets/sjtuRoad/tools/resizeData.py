import glob
import cv2 as cv

dataDirs = glob.glob('../data*')
size = (640, 384)

for dataDir in dataDirs:
    frameDir = dataDir + '/frame/'
    rgbPaths = (glob.glob(frameDir + '*'))

    depthDir = dataDir + '/depth/'
    depthPaths = (glob.glob(depthDir + '*'))

    for path in rgbPaths:
        img = cv.imread(path)
        img2 = cv.resize(img, size)
        # img2 = (cv.pyrDown(img))

        img2.dtype
        img2.shape
        # cv.imwrite(path, img2)

    for path in rgbPaths:
        img = cv.imread(path, -1)
        img2 = cv.resize(img, size)

        img2.dtype
        img2.shape
        # cv.imwrite(path, img2)
