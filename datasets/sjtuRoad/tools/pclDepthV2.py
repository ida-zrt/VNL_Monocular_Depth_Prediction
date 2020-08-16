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
pause = True
maxDepth = 280
outputSize = (2560, 768)
outputRoi = (0, 0, 2560, 768)

# 标定文件
calib_path = './calibfiles/20200622_122024_autoware_lidar_camera_calibration.yaml'
# extrisicFile = './calibResult.dat'
extrisicFile = './calibfiles/20200806_130338_lidar_cam_calib.yaml'

# 数据文件夹
dataDirs = glob.glob('../data*')
# 读取标定的信息
fs = cv.FileStorage(calib_path, cv.FILE_STORAGE_READ)
cameraMat = fs.getNode('CameraMat').mat()
distCoeff = fs.getNode('DistCoeff').mat()
fs.release()
if extrisicFile.endswith('.dat'):
    with open(extrisicFile, 'rb') as f:
        rvec, tvec = pickle.load(f)
else:
    fs = cv.FileStorage(extrisicFile, cv.FILE_STORAGE_READ)
    rvec = fs.getNode('rvec').mat()
    tvec = fs.getNode('tvec').mat()
    fs.release()

# 图像的大小, 不知道怎么从yaml中读取, 直接拷贝的
imgSize = (2560, 1536)
k = 0
for dataDir in dataDirs:
    if k == 27:
        break
    pcl_path = os.path.join(dataDir, 'pointCloud/')
    print('processing data in ' + pcl_path)
    clouds = glob.glob(pcl_path + '*')
    # 存放深度图像的文件夹
    depth_path = pcl_path + '../depth/'
    depthColoredPath = pcl_path + '../depthColored'
    img_path = pcl_path + '../pic/'
    compare_path = pcl_path + '../compare/'

    if not os.path.exists(depth_path):
        os.makedirs(depth_path)
    if not os.path.exists(img_path):
        os.makedirs(img_path)

    if save_depth_c:
        if not os.path.exists(depthColoredPath):
            os.makedirs(depthColoredPath)
        if not os.path.exists(compare_path):
            os.makedirs(compare_path)

    for cloud in clouds:
        cloudPts = pcl.load(cloud).to_array()
        depthImg = getDepthFromCloud(cloudPts, rvec, tvec, cameraMat,
                                     distCoeff, imgSize)
        coloredDepthImg = generateColoredDepthImg(depthImg)

        depthMask = np.zeros(imgSize[::-1]).astype(np.uint8)
        depthMask[depthImg != 0] = 255

        pic = cloud.replace('pointCloud', 'frame')
        pic = pic.replace('No.', 'frame')
        pic = pic.replace('.pcd', '.jpg')
        img = cv.imread(pic)
        img = cv.resize(img, imgSize)

        coloredDepthImgD = cv.dilate(
            coloredDepthImg, cv.getStructuringElement(cv.MORPH_ELLIPSE,
                                                      (3, 3)))
        compareImg = cv.addWeighted(img, 0.3, coloredDepthImgD, 1, 0)
        x, y, w, h = outputRoi
        compareImg = compareImg[y:y + h, x:x + w]

        if display_depth:
            print('Comparing depth and rgb files: {}'.format(
                os.path.split(cloud)[1]))

            cv.namedWindow('compare', cv.WINDOW_KEEPRATIO)
            cv.namedWindow('colored', cv.WINDOW_KEEPRATIO)
            cv.namedWindow('img', cv.WINDOW_KEEPRATIO)
            cv.namedWindow('depth', cv.WINDOW_KEEPRATIO)
            cv.namedWindow('mask', cv.WINDOW_KEEPRATIO)

            cv.imshow('compare', compareImg)
            cv.imshow('depth', depthImg)
            cv.imshow('colored', coloredDepthImg)
            cv.imshow('img', img)
            cv.imshow('mask', depthMask)

            k = cv.waitKey(int(not pause))
            if k == 27:
                break
            if k == ord('p'):
                pause = not pause

        fname = cloud.replace('pointCloud', 'depth')
        fname = fname.replace('pcd', 'png')
        fname = fname.replace('No.', 'depth')
        if save_depth:
            x, y, w, h = outputRoi
            img = img[y:y + h, x:x + w]
            img = cv.resize(img, outputSize)
            cv.imwrite(pic.replace('frame', 'pic'), img)
            depthImg = depthImg[y:y + h, x:x + w]
            depthImg = cv.resize(depthImg, outputSize)
            cv.imwrite(fname, depthImg)
            print("{} saved!!".format(fname))

        if save_depth_c:
            fname2 = fname.replace('depth', 'depthMask')
            cv.imwrite(fname2, depthMask)
            fname3 = fname.replace('depth', 'depthColored')
            cv.imwrite(fname3, coloredDepthImg)
            fname4 = fname.replace('depth', 'compare')
            cv.imwrite(fname4, compareImg)

cv.destroyAllWindows()
