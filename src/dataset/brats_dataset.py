import os
import math
import copy
import random

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from numpy import logical_and as l_and, logical_not as l_not
from scipy.spatial.distance import directed_hausdorff

import torch
from torch.utils.data import DataLoader
from torchvision import utils
from torch.utils.data import Dataset


import cv2
import nibabel as nib
import skimage.transform as skTrans

from config import BRATS_TRAIN_FOLDERS
from .utils import read_mri, RandomFlip, MinMaxNormalize, ScaleToFixed, ZeroChannel, ZeroSprinkle, ToTensor, Compose


class GeneralDataset(Dataset):

    def __init__(self,
                 metadata_df,
                 root_dir,
                 transform=None,
                 seg_transform=None,  ###
                 dataformat=None,  # indicates what shape (or content) should be returned (2D or 3D, etc.)
                 returndims=None,  # what size/shape 3D volumes should be returned as.
                 visualize=False,
                 modality=None,
                 pad=2,
                 device='cpu'):
        """
        Args:
            metadata_df (string): Path to the csv file w/ patient IDs
            root_dir (string): Directory for MR images
            transform (callable, optional)
        """
        self.device = device
        self.metadata_df = metadata_df
        self.root_dir = root_dir
        self.transform = transform
        self.seg_transform = seg_transform
        self.returndims = returndims
        self.modality = modality
        self.pad = pad

    def __len__(self):
        return len(self.metadata_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        BraTS20ID = self.metadata_df.iloc[idx].BraTS_2020_subject_ID

        # make dictonary of paths to MRI volumnes (modalities) and segmenation masks
        mr_path_dict = {}
        sequence_type = ['seg', 't1', 't1ce', 'flair', 't2']
        for seq in sequence_type:
            mr_path_dict[seq] = os.path.join(self.root_dir, BraTS20ID, BraTS20ID + '_' + seq + '.nii.gz')

        image, seg_image = read_mri(mr_path_dict=mr_path_dict, pad=self.pad)

        if seg_image is not None:
            seg_image[seg_image == 4] = 3

        if self.transform:
            image = self.transform(image)
        if self.seg_transform:
            seg_image = self.seg_transform(seg_image)
        else:
            print('no transform')
        print(image.shape)
        return (image, seg_image), BraTS20ID


def transformations(channels, resize_shape):
    # basic data augmentation
    prob_voxel_zero = 0  # 0.1
    prob_channel_zero = 0  # 0.5
    prob_true = 0  # 0.8

    randomflip = RandomFlip()

    # MRI transformations
    train_transformations = Compose([
        MinMaxNormalize(),
        ScaleToFixed((channels, resize_shape[0], resize_shape[1], resize_shape[2]),
                     interpolation=1,
                     channels=channels),
        ZeroSprinkle(prob_zero=prob_voxel_zero, prob_true=prob_true),
        ZeroChannel(prob_zero=prob_channel_zero),
        randomflip,
        ToTensor()
    ])

    # GT segmentation mask transformations

    seg_transformations = Compose([
        ScaleToFixed((1, resize_shape[0], resize_shape[1], resize_shape[2]),
                     interpolation=0,
                     channels=1),
        randomflip,
        ToTensor(),
    ])
    return train_transformations, seg_transformations


def get_datasets(channels, resize_shape, train_df, valid_df):

    train_transformations, seg_transformations = transformations(channels, resize_shape)

    transformed_dataset_train = GeneralDataset(metadata_df=train_df,
                                               root_dir=BRATS_TRAIN_FOLDERS,
                                               transform=train_transformations,
                                               seg_transform=seg_transformations,
                                               returndims=resize_shape)

    transformed_dataset_valid = GeneralDataset(metadata_df=valid_df,
                                               root_dir=BRATS_TRAIN_FOLDERS,
                                               transform=train_transformations,
                                               seg_transform=seg_transformations,
                                               returndims=resize_shape)

    return transformed_dataset_train, transformed_dataset_valid
