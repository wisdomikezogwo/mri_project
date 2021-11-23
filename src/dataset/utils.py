import random
import torch
import numpy as np
import nibabel as nib
import skimage.transform as skTrans


class ScaleToFixed(object):

    def __init__(self, new_shape, interpolation=1, channels=4):
        self.shape = new_shape
        self.interpolation = interpolation
        self.channels = channels

    def __call__(self, image):
        # print('first shape', image.shape)
        if image is not None:  # (some patients don't have segmentations)
            if self.channels == 1:
                short_shape = (self.shape[1], self.shape[2], self.shape[3])
                image = skTrans.resize(image, short_shape, order=self.interpolation, preserve_range=True)  #
                image = image.reshape(self.shape)
            else:
                image = skTrans.resize(image, self.shape, order=self.interpolation, preserve_range=True)  #

        return image


class RandomFlip(object):
    """Randomly flips (horizontally as well as vertically) the given PIL.Image with a probability of 0.5
    """
    def __init__(self, prob_flip=0.5):
        self.prob_flip = prob_flip

    def __call__(self, image):

        if random.random() < self.prob_flip:
            flip_type = np.random.randint(0, 3)  # flip across any 3D axis
            image = np.flip(image, flip_type)
        return image


class ZeroChannel(object):
    """Randomly sets channel to zero the given PIL.Image with a probability of 0.25
    """
    def __init__(self, prob_zero=0.25, channels=4):
        self.prob_zero = prob_zero
        self.channels = channels

    def __call__(self, image):

        if np.random.random() < self.prob_zero:
            channel_to_zero = np.random.randint(0, self.channels)  # flip across any 3D axis
            zeros = np.zeros((image.shape[1], image.shape[2], image.shape[3]))
            image[channel_to_zero, :, :, :] = zeros
        return image


class ZeroSprinkle(object):
    def __init__(self, prob_zero=0.25, prob_true=0.5, channels=4):
        self.prob_zero = prob_zero
        self.prob_true = prob_true
        self.channels = channels

    def __call__(self, image):

        if self.prob_true:
            mask = np.random.rand(image.shape[0], image.shape[1], image.shape[2], image.shape[3])
            mask[mask < self.prob_zero] = 0
            mask[mask > 0] = 1
            image = image*mask

        return image


class MinMaxNormalize(object):
    """Min-Max normalization
    """
    def __call__(self, image):
        def norm(im):
            im = im.astype(np.float32)
            min_v = np.min(im)
            max_v = np.max(im)
            im = (im - min_v)/(max_v - min_v)
            return im
        image = norm(image)
        return image


class ToTensor(object):
    def __init__(self, scale=1):
        self.scale = scale

    def __call__(self, image):
        if image is not None:
            image = image.astype(np.float32)
            image = image.reshape((image.shape[0], int(image.shape[1]/self.scale), int(image.shape[2]/self.scale),
                                   int(image.shape[3]/self.scale)))
            image_tensor = torch.from_numpy(image)
            return image_tensor
        else:
            return image


class Compose(object):
    """
    Composes several transforms together.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image):
        for i, t in enumerate(self.transforms):
            image = t(image)
        return image


def get_bb_3D(img, pad=0):
    '''
    This function returns a tumor 3D bounding box using a segmentation mask
    '''
    xs = np.nonzero(np.sum(np.sum(img, axis=1), axis=1))
    ys = np.nonzero(np.sum(np.sum(img, axis=0), axis=1))
    zs = np.nonzero(np.sum(np.sum(img, axis=0), axis=0))
    xmin, xmax = np.min(xs), np.max(xs)
    ymin, ymax = np.min(ys), np.max(ys)
    zmin, zmax = np.min(zs), np.max(zs)
    bbox = (xmin-pad, ymin-pad, zmin-pad, xmax+pad, ymax+pad, zmax+pad)
    return bbox


def min_max(img):
    '''
    Min-max normalization
    '''
    return (img - img.min()) / (img.max() - img.min())


def read_mri(mr_path_dict, pad=0):

    image_shape = nib.load(mr_path_dict['flair']).get_fdata().shape
    bb_seg = get_bb_3D(nib.load(mr_path_dict['flair']).get_fdata())
    (xmin, ymin, zmin, xmax, ymax, zmax) = bb_seg

    xmin = np.max([0, xmin-pad])
    ymin = np.max([0, ymin-pad])
    zmin = np.max([0, zmin-pad])

    xmax = np.min([image_shape[0]-1, xmax+pad])
    ymax = np.min([image_shape[1]-1, ymax+pad])
    zmax = np.min([image_shape[2]-1, zmax+pad])

    img_dict = {}
    for key in ['flair', 't1', 't1ce', 't2', 'seg']:
        img = nib.load(mr_path_dict[key])
        img_data = img.get_fdata()
        img_dict[key] = img_data[xmin:xmax, ymin:ymax, zmin:zmax]

    stacked_img = np.stack([min_max(img_dict['flair']), min_max(img_dict['t1']),min_max(img_dict['t1ce']),min_max(img_dict['t2'])], axis=0)
    return stacked_img, img_dict['seg']

