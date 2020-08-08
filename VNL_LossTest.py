from lib.models.VNL_loss import VNL_Loss

import torch
import numpy as np
import cv2 as cv

fx, fy = [2.1624e+03, 2.1531e+03]
optc = [(1.25654e+03), (7.719e+02)]
testSample = './datasets/sjtuRoad/data_2020_07_29_17_44_03/depth/depth0001.png'
img1 = cv.imread(testSample, -1).astype(float) / 65536

vnl_loss = VNL_Loss(fx,
                    fy,
                    img1.shape,
                    optc,
                    delta_diff_x=5 / 280,
                    delta_diff_y=5 / 280,
                    delta_diff_z=5 / 280,
                    z_thres=0,
                    sample_ratio=0.25)
img1 = img1.reshape((1, 1, img1.shape[0], img1.shape[1]))
img2 = np.random.random(size=img1.shape)

img1 = torch.tensor(img1.astype(np.float32)).cuda()
img2 = torch.tensor(img2.astype(np.float32)).cuda()

loss = vnl_loss(img1, img2)
print(loss)

# img1eg = cv.imread('./0000000005.png', -1).astype(float) / 22000
# img2eg = cv.imread('./0000000006.png', -1).astype(float) / 22000

# vnl_loss2 = VNL_Loss(fx, fy, img1eg.shape, img1eg.shape[::-1])

# img1eg = img1eg.reshape((1, 1, img1eg.shape[0], img1eg.shape[1]))
# img2eg = img2eg.reshape((1, 1, img2eg.shape[0], img2eg.shape[1]))
# img1eg = torch.tensor(img1eg.astype(np.float32)).cuda()
# img2eg = torch.tensor(img2eg.astype(np.float32)).cuda()

# loss = vnl_loss2(img1eg, img2eg)
# print(loss)
