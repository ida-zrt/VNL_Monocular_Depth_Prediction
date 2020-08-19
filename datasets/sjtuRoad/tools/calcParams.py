#!/usr/bin/python
# -- coding:UTF-8
import cv2 as cv
import pcl
import glob
import os
import pickle
from utils.depthConversion import *
import numpy as np

# global settings
display_depth = False
save_depth = True
save_depth_c = False
pause = False
maxDepth = 280
outputSize = (2560, 768)

mean = np.zeros(3)
var = np.zeros(3)
maxDepth = 0

# 数据文件夹
dataDirs = glob.glob('../data*')
imgCount = 0

for dataDir in dataDirs:
    rgbPath = os.path.join(dataDir, 'pic/')
    imgs = glob.glob(rgbPath + '*')
    imgCount = imgCount + len(imgs)
    for fname in imgs:
        img = cv.imread(fname)
        b, g, r = cv.split(img)
        mean[0] = mean[0] + np.mean(r)
        mean[1] = mean[1] + np.mean(g)
        mean[2] = mean[2] + np.mean(b)
        var[0] = var[0] + np.var(r / 255)
        var[1] = var[1] + np.var(g / 255)
        var[2] = var[2] + np.var(b / 255)

    depthPath = os.path.join(dataDir, 'depth/')
    depthImgs = glob.glob(depthPath + '*')
    for fname in depthImgs:
        depth = cv.imread(fname, -1)
        if np.max(depth) > maxDepth:
            maxDepth = np.max(depth)

# rgb mean and var
mean = mean / imgCount / 255
var = var / imgCount * 255

print('RGB Mean')
print(mean)
print('Var')
print(var)
print('MaxDepth')
print(maxDepth)