# a class for generate depth image
# -- coding:UTF-8
import cv2 as cv
import numpy as np
import os
import glob
from depthConversion import getDepthFromCloud, generateColoredDepthImg
from depthInterp import interpDepthImg
import pcl
import shutil


class depthGenterator():
    def __init__(self, pclDir, maxDepth):
        self.maxDepth = maxDepth
        self.pclDir = pclDir
        if not self.pclDir.endswith(os.sep):
            self.pclDir += os.sep
        self.pclDir += 'pointCloud' + os.sep
        self.getCalibrationInfo()
        self.makeSaveDirs()

    def makeSaveDirs(self):
        # 存放深度图像的文件夹
        depth_path = self.pclDir + '../depth/'
        depthColoredPath = self.pclDir + '../depthColored'
        img_path = self.pclDir + '../pic/'
        compare_path = self.pclDir + '../compare/'
        depth_raw = self.pclDir + '../depthRaw/'

        paths = [
            depth_path, depthColoredPath, img_path, compare_path, depth_raw
        ]
        # create dirs
        for path in paths:
            if not os.path.exists(path):
                os.makedirs(path)

    def clear(self, all=False):
        depthColoredPath = self.pclDir + '../depthColored'
        compare_path = self.pclDir + '../compare/'
        shutil.rmtree(depthColoredPath)
        shutil.rmtree(compare_path)

        if all:
            depth_path = self.pclDir + '../depth/'
            img_path = self.pclDir + '../pic/'
            shutil.rmtree(depth_path)
            shutil.rmtree(img_path)

    def getCalibrationInfo(self):
        calib_path = self.pclDir + '../calib.yaml'
        if os.path.exists(calib_path):
            fs = cv.FileStorage(calib_path, cv.FILE_STORAGE_READ)
            self.cameraMat = fs.getNode('CameraMat').mat()
            self.distCoeff = fs.getNode('DistCoeff').mat()
            self.rvec = fs.getNode('rvec').mat()
            self.tvec = fs.getNode('tvec').mat()
            self.imgSize = np.squeeze(fs.getNode('ImageSize').mat())
            fs.release()
        else:
            print('Calibration file not found at {}'.format(calib_path))

    def saveDepth(self,
                  interp=True,
                  coloeredCompare=False,
                  saveRaw=True,
                  **kwargs):
        # kwargs:
        # outputROI: output roi. Default None
        # outputSize: output size, crop before resize!! Default None
        # method: method used for interpolation, 'linear', 'cubic', 'nearest'. Default linear
        # ksize: kernel size for fitting local plain. Default 0
        self.makeSaveDirs()
        outputRoi = kwargs.get('outputRoi', None)
        outputSize = kwargs.get('outputSize', None)
        clouds = glob.glob(self.pclDir + '*')
        for cloud in clouds:
            # get picture path
            pic = cloud.replace('pointCloud', 'frame')
            pic = pic.replace('No.', 'frame')
            pic = pic.replace('.pcd', '.jpg')
            # get Depth save Path
            fname = cloud.replace('pointCloud', 'depth')
            fname = fname.replace('pcd', 'png')
            fname = fname.replace('No.', 'depth')

            img = cv.imread(pic)
            img = cv.resize(img, (self.imgSize[0], self.imgSize[1]))

            cloudPts = pcl.load(cloud).to_array()
            depthRaw = getDepthFromCloud(cloudPts, self.rvec, self.tvec,
                                         self.cameraMat, self.distCoeff,
                                         self.imgSize, self.maxDepth)

            # save depth data generated from lidar points
            if saveRaw:
                x, y, w, h = outputRoi
                depthRawSave = depthRaw[y:y + h, x:x + w]
                depthRawSave = cv.resize(depthRawSave, outputSize)
                fnameRaw = fname.replace('depth', 'depthRaw')
                cv.imwrite(fnameRaw, depthRawSave.astype(np.uint16))
                print("{} saved!!".format(fnameRaw))

            depthToSave = depthRaw
            x, y, w, h = outputRoi
            depthToSave = depthToSave[y:y + h, x:x + w]

            # interp the depth data from lidar point depth
            if interp:
                ksize = kwargs.get('ksize', 0)
                method = kwargs.get('method', 'linear')
                depthToSave = interpDepthImg(depthToSave, ksize, method)

            if outputRoi is None or outputSize is None:
                pass
            else:
                x, y, w, h = outputRoi
                imgS = img[y:y + h, x:x + w]
                imgS = cv.resize(imgS, outputSize)
                depthToSave = depthToSave[y:y + h, x:x + w]
                depthToSave = cv.resize(depthToSave, outputSize)
            # save depth and train image
            cv.imwrite(fname, depthToSave.astype(np.uint16))
            cv.imwrite(pic.replace('frame', 'pic'), imgS)
            print("{} saved!!".format(fname))

            if coloeredCompare:
                coloredDepthImg = generateColoredDepthImg(depthRaw)
                # generate Image + colored Depth for comparesion
                coloredDepthImgD = cv.dilate(
                    coloredDepthImg,
                    cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3)))
                compareImg = cv.addWeighted(img, 0.3, coloredDepthImgD, 1, 0)
                if outputRoi is None or outputSize is None:
                    pass
                else:
                    x, y, w, h = outputRoi
                    compareImg = compareImg[y:y + h, x:x + w]

                fname3 = fname.replace('depth', 'depthColored')
                cv.imwrite(fname3, coloredDepthImg)
                fname4 = fname.replace('depth', 'compare')
                cv.imwrite(fname4, compareImg)
