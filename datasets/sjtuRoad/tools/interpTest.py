import cv2 as cv
from utils.depthInterp import interpDepthImg
import numpy as np
import scipy

img = cv.imread('./depthRaw0002.png', -1)

methods = ['cubic', 'linear', 'nearest']
ksize = (5, 8)
results = [
    cv.imwrite('./testResults/x{}y{}_{}_test.png'.format(ksize[0], ksize[1], x),
               interpDepthImg(img, ksize, x).astype(np.uint16))
    for x in methods
]
print(results)
