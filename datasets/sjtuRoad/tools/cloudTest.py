from utils.depth2pc import depth2points2
import cv2 as cv
from utils.visualize import showPoints
import mayavi.mlab as mlab
import numpy as np

# depthFile = './depthRaw0002.png'
depthFile = './testResults/x3y3_nearest_test.png'

depthImg = cv.imread(depthFile, -1)

pc = depth2points2(depthImg, './calibfiles/calib.yaml', 280)

fig = mlab.figure(bgcolor=(0, 0, 0))

showPoints(pc, fig)

mlab.show()
