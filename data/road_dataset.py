import cv2
import torch
import os.path
import numpy as np
import scipy.io as sio
from lib.core.config import cfg
import torchvision.transforms as transforms
from lib.utils.logging import setup_logging

logger = setup_logging(__name__)


class roadDataset():
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.maxDepth = 65536
        self.dir_anno = os.path.join(cfg.ROOT_DIR, opt.dataroot, 'annotations',
                                     opt.phase_anno + '.txt')
        self.getDataPaths()
        self.uniform_size = (768, 2560)

    def getDataPaths(self):
        with open(self.dir_anno, 'r') as load_f:
            lines = load_f.readlines()
        lines = [line.strip() for line in lines]
        self.rgbPath = [x.split('"')[1] for x in lines]
        self.depthPath = [x.split('"')[3] for x in lines]

    def __getitem__(self, anno_index):
        data = self.online_aug(anno_index)
        return data

    def online_aug(self, anno_index):
        """
        Augment data for training online randomly. The invalid parts in the depth map are set to -1.0, while the parts
        in depth bins are set to cfg.MODEL.DECODER_OUTPUT_C + 1.
        :param anno_index: data index.
        """
        A_path = self.rgbPath[anno_index]
        B_path = self.depthPath[anno_index]

        A = cv2.imread(A_path)  # bgr, H*W*C
        B = cv2.imread(B_path, -1) / self.maxDepth
        flip_flg, crop_size, pad, resize_ratio = self.set_flip_pad_reshape_crop(
        )

        A_resize = self.flip_pad_reshape_crop(A, flip_flg, crop_size, pad, 128)
        B_resize = self.flip_pad_reshape_crop(B, flip_flg, crop_size, pad, -1)

        A_resize = A_resize.transpose((2, 0, 1))
        B_resize = B_resize[np.newaxis, :, :]

        # change the color channel, bgr -> rgb
        A_resize = A_resize[::-1, :, :]

        # to torch, normalize
        A_resize = self.scale_torch(A_resize, 255.)
        B_resize = self.scale_torch(B_resize, resize_ratio)

        B_bins = self.depth_to_bins(B_resize)
        invalid_side = [int(pad[0] * resize_ratio), 0, 0, 0]

        # A: rgb files, B: depth files
        data = {
            'A': A_resize,
            'B': B_resize,
            'A_raw': A,
            'B_raw': B,
            'B_bins': B_bins,
            'A_paths': A_path,
            'B_paths': B_path,
            'invalid_side': np.array(invalid_side),
            'ratio': np.float32(1.0 / resize_ratio)
        }
        return data

    def set_flip_pad_reshape_crop(self):
        """
        Set flip, padding, reshaping, and cropping factors for the image.
        :return:
        """
        # flip
        flip_prob = np.random.uniform(0.0, 1.0)
        flip_flg = True if flip_prob > 0.5 and 'train' in self.opt.phase else False

        # raw_size = np.array(
        #     [cfg.DATASET.CROP_SIZE[1], 416, 448, 480, 512, 544, 576, 608, 640])
        raw_size = np.array(
            [cfg.DATASET.CROP_SIZE[1], 672, 704, 736, 608, 640])
        size_index = np.random.randint(0,
                                       6) if 'train' in self.opt.phase else -1

        # pad
        pad_height = raw_size[size_index] - self.uniform_size[0] if raw_size[size_index] > self.uniform_size[0]\
                    else 0
        pad = [pad_height, 0, 0, 0]  # [up, down, left, right]

        # crop
        crop_height = raw_size[size_index]
        crop_width = raw_size[size_index]
        start_x = np.random.randint(0,
                                    int(self.uniform_size[1] - crop_width) + 1)
        start_y = 0 if pad_height != 0 else np.random.randint(
            0,
            int(self.uniform_size[0] - crop_height) + 1)
        crop_size = [start_x, start_y, crop_height, crop_width]

        resize_ratio = float(cfg.DATASET.CROP_SIZE[1] / crop_width)

        return flip_flg, crop_size, pad, resize_ratio

    def flip_pad_reshape_crop(self, img, flip, crop_size, pad, pad_value=0):
        """
        Flip, pad, reshape, and crop the image.
        :param img: input image, [C, H, W]
        :param flip: flip flag
        :param crop_size: crop size for the image, [x, y, width, height]
        :param pad: pad the image, [up, down, left, right]
        :param pad_value: padding value
        :return:
        """
        # Flip
        if flip:
            img = np.flip(img, axis=1)

        # Pad the raw image
        if len(img.shape) == 3:
            img_pad = np.pad(img, ((pad[0], pad[1]), (pad[2], pad[3]), (0, 0)),
                             'constant',
                             constant_values=(pad_value, pad_value))
        else:
            img_pad = np.pad(img, ((pad[0], pad[1]), (pad[2], pad[3])),
                             'constant',
                             constant_values=(pad_value, pad_value))
        # Crop the resized image
        img_crop = img_pad[crop_size[1]:crop_size[1] + crop_size[3],
                           crop_size[0]:crop_size[0] + crop_size[2]]

        # Resize the raw image
        img_resize = cv2.resize(
            img_crop, (cfg.DATASET.CROP_SIZE[1], cfg.DATASET.CROP_SIZE[0]),
            interpolation=cv2.INTER_LINEAR)
        return img_resize

    def depth_to_bins(self, depth):
        """
        Discretize depth into depth bins
        Mark invalid padding area as cfg.MODEL.DECODER_OUTPUT_C + 1
        :param depth: 1-channel depth, [1, h, w]
        :return: depth bins [1, h, w]
        """
        invalid_mask = depth <= 0.
        depth[depth < cfg.DATASET.DEPTH_MIN] = cfg.DATASET.DEPTH_MIN
        depth[depth > cfg.DATASET.DEPTH_MAX] = cfg.DATASET.DEPTH_MAX
        bins = ((torch.log10(depth) - cfg.DATASET.DEPTH_MIN_LOG) /
                cfg.DATASET.DEPTH_BIN_INTERVAL).to(torch.int)
        bins[invalid_mask] = cfg.MODEL.DECODER_OUTPUT_C + 1
        bins[bins ==
             cfg.MODEL.DECODER_OUTPUT_C] = cfg.MODEL.DECODER_OUTPUT_C - 1
        depth[invalid_mask] = -1.0
        return bins

    def scale_torch(self, img, scale):
        """
        Scale the image and output it in torch.tensor.
        :param img: input image. [C, H, W]
        :param scale: the scale factor. float
        :return: img. [C, H, W
        """
        img = img.astype(np.float32)
        img /= scale
        img = torch.from_numpy(img.copy())
        if img.size(0) == 3:
            img = transforms.Normalize(cfg.DATASET.RGB_PIXEL_MEANS,
                                       cfg.DATASET.RGB_PIXEL_VARS)(img)
        else:
            img = transforms.Normalize((0, ), (1, ))(img)
        return img

    def __len__(self):
        return len(self.depthPath)

    def name(self):
        return 'NYUDV2'
