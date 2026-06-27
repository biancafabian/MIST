#!/usr/bin/env python
# -*- coding:utf-8 -*-

import os
import random
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
from torch.utils.data import Dataset


def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


def random_zoom(image, label, zoom_range=(0.85, 1.25)):
    factor = np.random.uniform(zoom_range[0], zoom_range[1])
    h, w = image.shape
    image_z = zoom(image, factor, order=3)
    label_z = zoom(label, factor, order=0)
    zh, zw = image_z.shape
    if factor > 1.0:
        # zoomed in: crop center back to original size
        sh = (zh - h) // 2
        sw = (zw - w) // 2
        image = image_z[sh:sh + h, sw:sw + w]
        label = label_z[sh:sh + h, sw:sw + w]
    else:
        # zoomed out: pad back to original size
        ph0 = (h - zh) // 2
        ph1 = h - zh - ph0
        pw0 = (w - zw) // 2
        pw1 = w - zw - pw0
        image = np.pad(image_z, ((ph0, ph1), (pw0, pw1)), mode='constant', constant_values=0)
        label = np.pad(label_z, ((ph0, ph1), (pw0, pw1)), mode='constant', constant_values=0)
    return image, label


def random_shift(image, label, max_frac=0.1):
    h, w = image.shape
    sh = int(np.random.uniform(-max_frac, max_frac) * h)
    sw = int(np.random.uniform(-max_frac, max_frac) * w)
    image = ndimage.shift(image, [sh, sw], order=3, mode='constant', cval=0)
    label = ndimage.shift(label, [sh, sw], order=0, mode='constant', cval=0)
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        # rotation/flip: uniform over three outcomes so each has ~33% probability
        r = random.random()
        if r < 0.33:
            image, label = random_rot_flip(image, label)
        elif r < 0.66:
            image, label = random_rotate(image, label)

        # zoom and shift applied independently with 50% probability each
        if random.random() > 0.5:
            image, label = random_zoom(image, label)
        if random.random() > 0.5:
            image, label = random_shift(image, label)

        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=3)
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32))
        sample = {'image': image, 'label': label.long()}
        return sample


class ACDCdataset(Dataset):
    def __init__(self, base_dir, list_dir, split, transform=None):
        self.transform = transform  # using transform in torch!
        self.split = split
        self.sample_list = open(os.path.join(list_dir, self.split+'.txt')).readlines()
        self.data_dir = base_dir

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        if self.split == "train" or self.split == "valid":
            slice_name = self.sample_list[idx].strip('\n')
            data_path = os.path.join(self.data_dir, self.split, slice_name)
            data = np.load(data_path)
            image, label = data['img'], data['label']
        else:
            vol_name = self.sample_list[idx].strip('\n')
            filepath = self.data_dir + "/{}".format(vol_name)
            data = np.load(filepath)
            image, label = data['img'], data['label']

        sample = {'image': image, 'label': label}
        if self.transform and self.split == "train":
            sample = self.transform(sample)
        sample['case_name'] = self.sample_list[idx].strip('\n')
        return sample
