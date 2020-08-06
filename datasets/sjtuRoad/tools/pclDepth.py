#!/usr/bin/python
# -- coding:UTF-8
import cv2 as cv
import numpy as np
import pcl
import glob
import os

# global settings
use_undistort = True
display_depth = False
save_depth = False
save_depth_c = False
compareRGBD = True
maxDepth = 280

# outputSize = (640, 384)
# outputSize = (1280, 768)
outputSize = (2560, 1536)

# 设置标定文件, pcd文件的路径
# calib_path = './20200804_125204_autoware_lidar_camera_calibration.yaml'
# calib_path = './20200804_132052_autoware_lidar_camera_calibration.yaml'
calib_path = './20200622_122024_autoware_lidar_camera_calibration.yaml'

pcl_path = '../data_2020_07_31_10_46_48/pointCloud/'
# pcl_path = '../data_2020_07_29_17_44_03/pointCloud/'

clouds = glob.glob(pcl_path + '*')
# 存放深度图像的文件夹
depth_path = pcl_path + '../depth/'
depthMaskPath = pcl_path + '../depthMask'
depthColoredPath = pcl_path + '../depthColored'
img_path = pcl_path + '../pic/'
compare_path = pcl_path + '../compare/'

if not os.path.exists(depth_path):
    os.makedirs(depth_path)

if not os.path.exists(img_path):
    os.makedirs(img_path)

if save_depth_c:
    if not os.path.exists(depthMaskPath):
        os.makedirs(depthMaskPath)
    if not os.path.exists(depthColoredPath):
        os.makedirs(depthColoredPath)
if compareRGBD:
    if not os.path.exists(compare_path):
        os.makedirs(compare_path)

# 读取标定的信息
fs = cv.FileStorage(calib_path, cv.FILE_STORAGE_READ)
lidar2camMat = np.linalg.inv(fs.getNode('CameraExtrinsicMat').mat())
# lidar2camMat = fs.getNode('CameraExtrinsicMat').mat()
cameraMat = fs.getNode('CameraMat').mat()
distCoeff = fs.getNode('DistCoeff').mat()

# 图像的大小, 不知道怎么从yaml中读取, 直接拷贝的
imgSize = (2560, 1536)
roi = (0, 0, 2560, 1536)

# 消除相机畸变
if use_undistort:
    w, h = imgSize[:2]
    NewCameraMat, roi = cv.getOptimalNewCameraMatrix(cameraMat, distCoeff,
                                                     (w, h), 1, (w, h))
    mapx, mapy = cv.initUndistortRectifyMap(cameraMat, distCoeff, None,
                                            NewCameraMat, (w, h), 5)
    # cameraMat = NewCameraMat

for cloud in clouds:
    # 将激光点云转换到相机的视角下
    cloudPts = pcl.load(cloud).to_array()
    cloudPts = np.concatenate(
        [cloudPts, np.ones((cloudPts.shape[0], 1))], axis=1).T
    newCloudPts = np.matmul(lidar2camMat, cloudPts)[:3, :]

    # 利用相机矩阵计算像素坐标
    pixelPts = (np.matmul(cameraMat, newCloudPts) /
                newCloudPts[-1, :]).astype(int)[:2, :]

    # 只取在相机内部的点
    x_inRange = np.logical_and(pixelPts[0] >= 0, pixelPts[0] < imgSize[0])
    y_inRange = np.logical_and(pixelPts[1] >= 0, pixelPts[1] < imgSize[1])
    inRange = np.logical_and(x_inRange, y_inRange)
    pixel = pixelPts[:, inRange]
    depth = newCloudPts[-1, :]

    # 将depth的值填入对应的像素位置
    depthImg = np.zeros(imgSize)
    depthImg[pixel[0], pixel[1]] = depth[inRange]
    # 之前图像用的是[x,y,channel], 正常的图像是[y,x,channel], 因此需要转置一下
    depthImg = (np.transpose(depthImg) * 65536 / maxDepth).astype(np.uint16)

    # 生成一个有颜色的深度图像, 颜色的Hue值随深度改变
    if compareRGBD or save_depth_c:
        depthHue = (depth[inRange] / maxDepth * 90).astype(np.uint8) + 90
        hsv = np.concatenate([
            np.expand_dims(depthHue, 1),
            np.ones((depthHue.shape[0], 2)) * 230
        ], 1)

        coloredDepthImg = np.zeros([imgSize[0], imgSize[1], 3])
        coloredDepthImg[pixel[0], pixel[1], :] = hsv
        coloredDepthImg = np.transpose(coloredDepthImg,
                                       (1, 0, 2)).astype(np.uint8)
        coloredDepthImg = cv.cvtColor(coloredDepthImg, cv.COLOR_HSV2BGR)

        # 生成一个mask, 白色的点代表这个上面有深度信息
        depthMask = np.zeros(imgSize)
        depthMask[depthImg.T != 0] = 255
        depthMask = depthMask.T

        coloredDepthImg = cv.dilate(
            coloredDepthImg, cv.getStructuringElement(cv.MORPH_ELLIPSE,
                                                      (3, 3)))
        # 将深度图与原图像相加进行对比
        if compareRGBD:
            pic = cloud.replace('pointCloud', 'frame')
            pic = pic.replace('No.', 'frame')
            pic = pic.replace('.pcd', '.jpg')

            img = cv.imread(pic)
            img = cv.resize(img, imgSize)
            if use_undistort:
                img = cv.remap(img, mapx, mapy, cv.INTER_CUBIC)
                img = img[roi[1]:roi[1] + roi[3], roi[0]:roi[0] + roi[2]]
                coloredDepthImg = coloredDepthImg[roi[1]:roi[1] + roi[3],
                                                  roi[0]:roi[0] + roi[2]]
                depthImg = depthImg[roi[1]:roi[1] + roi[3],
                                    roi[0]:roi[0] + roi[2]]
                depthMask = depthMask[roi[1]:roi[1] + roi[3],
                                      roi[0]:roi[0] + roi[2]]

            img = cv.resize(img, outputSize)
            coloredDepthImg = cv.resize(coloredDepthImg, outputSize)

            compare = cv.addWeighted(img, 0.3, coloredDepthImg, 1, 0)
            print('Comparing depth and rgb files: {}'.format(
                os.path.split(cloud)[1]))
            cv.namedWindow('compare', cv.WINDOW_KEEPRATIO)
            cv.namedWindow('depth_colored', cv.WINDOW_KEEPRATIO)
            cv.namedWindow('img', cv.WINDOW_KEEPRATIO)
            cv.imshow('compare', compare)
            cv.imshow('depth_colored', coloredDepthImg)
            cv.imshow('img', img)
            k = cv.waitKey(0)
            if k == 27:
                break

    if display_depth:
        cv.namedWindow('depth', cv.WINDOW_KEEPRATIO)
        cv.imshow('depth', (depthImg))
        cv.waitKey(1)

    if save_depth:
        pic = cloud.replace('pointCloud', 'frame')
        pic = pic.replace('No.', 'frame')
        pic = pic.replace('.pcd', '.jpg')

        img = cv.imread(pic)
        img = cv.resize(img, outputSize)
        cv.imwrite(pic.replace('frame', 'pic'), img)
        fname = cloud.replace('pointCloud', 'depth')
        fname = fname.replace('pcd', 'png')
        fname = fname.replace('No.', 'depth')
        depthImg = cv.resize(depthImg, outputSize)
        cv.imwrite(fname, depthImg)
        print("{} saved!!".format(fname))

    if save_depth_c and save_depth:
        fname2 = fname.replace('depth', 'depthMask')
        cv.imwrite(fname2, depthMask)
        fname3 = fname.replace('depth', 'depthColored')
        cv.imwrite(fname3, coloredDepthImg)
        if compareRGBD:
            fname4 = fname.replace('depth', 'compare')
            cv.imwrite(fname4, compare)

cv.destroyAllWindows()
