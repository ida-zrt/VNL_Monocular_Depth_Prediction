#!/usr/bin/python
# -- coding:UTF-8
import cv2 as cv
import pcl
import glob
import os
import pickle
from utils.depthConversion import *
import numpy as np
from utils.depthGenerator import depthGenterator as dG

# global settings
display_depth = False
save_depth = True
save_depth_c = False
pause = True
maxDepth = 280
outputSize = (2560, 768)
outputRoi = (0, 0, 2560, 768)

# 数据文件夹
dataDirs = glob.glob('../data*')[::-1]

for dataDir in dataDirs:
    generator = dG(dataDir, 200)
    generator.clear(all=True)
    generator.saveDepth(coloeredCompare=False,
                        outputRoi=outputRoi,
                        outputSize=outputSize,
                        method='nearest')
