import os
import cv2
import glob
import torch
import random
import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms

from option import args
from utils import read_MEF, read_MEF_simple


class TrainMEF(data.Dataset):
    '''
    Using MEF dataset to train
    '''

    def __init__(self):
        super(TrainMEF, self).__init__()
        self.device = torch.device(args.device)
        self.over_name, self.under_name = read_MEF_simple(args.data_path)
        self.patch = args.patch

    def __len__(self):
        assert len(self.over_name) == len(self.under_name)
        return len(self.over_name)

    def __getitem__(self, idx):
        # Get image with gray scale
        img_over = cv2.imread(self.over_name[idx], cv2.IMREAD_COLOR)
        img_under = cv2.imread(self.under_name[idx], cv2.IMREAD_COLOR)

        # convert image from BGR to YCbCr
        img_over = cv2.cvtColor(img_over, cv2.COLOR_BGR2YCrCb)
        img_under = cv2.cvtColor(img_under, cv2.COLOR_BGR2YCrCb)
        img_over = np.array(img_over, dtype='float32') / 255.0
        img_under = np.array(img_under, dtype='float32') / 255.0

        # crop and augmentation
        img_over, img_under = self.crops(img_over, img_under)

        # extract Y channel of visible image
        img_over_Y = img_over[:, :, 0:1]
        img_under_Y = img_under[:, :, 0:1]

        # Permute the images to tensor format
        img_over_Y = np.transpose(img_over_Y, (2, 0, 1))
        img_under_Y = np.transpose(img_under_Y, (2, 0, 1))

        # Return image
        self.input_over_Y = img_over_Y.copy()
        self.input_under_Y = img_under_Y.copy()

        return self.input_over_Y, self.input_under_Y

    def crops(self, train_img_over, train_img_under):
        # Take random crops
        h, w, _ = train_img_over.shape
        x = random.randint(0, h - self.patch)
        y = random.randint(0, w - self.patch)
        train_img_over = train_img_over[x: x + self.patch, y: y + self.patch, :]
        train_img_under = train_img_under[x: x + self.patch, y: y + self.patch, :]

        return train_img_over, train_img_under


class TestMEF(data.Dataset):
    '''
    Using MEF dataset to test
    '''

    def __init__(self):
        super(TestMEF, self).__init__()
        self.device = torch.device(args.device)
        self.over_name, self.under_name = read_MEF_simple(args.data_test_path)

    def __len__(self):
        assert len(self.over_name) == len(self.under_name)
        return len(self.over_name)

    def __getitem__(self, idx):
        # Get image with gray scale
        img_over = cv2.imread(self.over_name[idx], cv2.IMREAD_COLOR)
        img_under = cv2.imread(self.under_name[idx], cv2.IMREAD_COLOR)

        # resize the test image
        H, W, _ = img_over.shape
        newH, newW = round(H / args.resize), round(W / args.resize)
        img_over = cv2.resize(img_over, (newW, newH))
        img_under = cv2.resize(img_under, (newW, newH))

        # convert visible image from BGR to YCbCr
        img_over = cv2.cvtColor(img_over, cv2.COLOR_BGR2YCrCb)
        img_under = cv2.cvtColor(img_under, cv2.COLOR_BGR2YCrCb)
        img_over = np.array(img_over, dtype='float32') / 255.0
        img_under = np.array(img_under, dtype='float32') / 255.0

        # extract Y, Cb, Cr channel of visible image
        img_over_Y = img_over[:, :, 0:1]
        img_over_Cb = img_over[:, :, 2:3]
        img_over_Cr = img_over[:, :, 1:2]
        img_under_Y = img_under[:, :, 0:1]
        img_under_Cb = img_under[:, :, 2:3]
        img_under_Cr = img_under[:, :, 1:2]

        # Permute the images to tensor format
        img_over_Y = np.transpose(img_over_Y, (2, 0, 1))
        img_under_Y = np.transpose(img_under_Y, (2, 0, 1))

        # Return image
        self.img_over_Y = img_over_Y.copy()
        self.img_under_Y = img_under_Y.copy()

        return self.img_over_Y, self.img_under_Y, img_over_Cb, img_over_Cr, img_under_Cb, img_under_Cr, self.over_name[idx]

