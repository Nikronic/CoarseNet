from __future__ import print_function, division
from PIL import Image
from skimage import feature, color
import numpy as np
import random

import tarfile
import io
import pandas as pd

from torch.utils.data import Dataset

from utils.Halftoning.halftone import generate_halftone


class PlacesDataset(Dataset):
    def __init__(self, txt_path='data/filelist.txt', img_dir='data.tar', transform=None):
        """
                Initialize data set as a list of IDs corresponding to each item of data set

                :param img_dir: path to image files as a uncompressed tar archive
                :param txt_path: a text file containing names of all of images line by line
                :param transform: apply some transforms like cropping, rotating, etc on input image

                :return a 3-value dict containing input image (y_descreen) as ground truth, input image X as halftone image
                        and edge-map (y_edge) of ground truth image to feed into the network.
                """

        df = pd.read_csv(txt_path, sep=' ', index_col=0)
        self.img_names = df.index.values
        self.txt_path = txt_path
        self.img_dir = img_dir
        self.transform = transform

    def get_image_by_name(self, name):
        """
        Gets a image by a name gathered from file list csv file

        :param name: name of targeted image
        :return: a PIL image
        """
        with tarfile.open(self.img_dir) as tf:
            tarinfo = tf.getmember(name)
            image = tf.extractfile(tarinfo)
            image = image.read()
            image = Image.open(io.BytesIO(image))
        return image

    def __len__(self):
        """
        Return the length of data set using list of IDs

        :return: number of samples in data set
        """
        return len(self.img_names)

    def __getitem__(self, index):
        """
        Generate one item of data set. Here we apply our preprocessing things like halftone styles and
        subtractive color process using CMYK color model, generating edge-maps, etc.

        :param index: index of item in IDs list

        :return: a sample of data as a dict
        """

        y_descreen = self.get_image_by_name(self.img_names[index])

        # generate halftone image
        X = generate_halftone(y_descreen)

        # generate edge-map
        y_edge = self.canny_edge_detector(y_descreen)

        if self.transform is not None:
            X = self.transform(X)
            y_descreen = self.transform(y_descreen)
            y_edge = self.transform(y_edge)

        sample = {'X': X,
                  'y_descreen': y_descreen,
                  'y_edge': y_edge}

        return sample

    @staticmethod
    def canny_edge_detector(image):
        """
        Returns a binary image with same size of source image which each pixel determines belonging to an edge or not.

        :param image: PIL image
        :return: Binary numpy array
        """
        image = np.array(image)
        image = color.rgb2grey(image)
        edges = feature.canny(image, sigma=1)  # TODO: the sigma hyper parameter value is not defined in the paper.
        edges = edges.astype(float)  # torch tensors need float
        edges = Image.fromarray(edges)
        return edges


# https://discuss.pytorch.org/t/adding-gaussion-noise-in-cifar10-dataset/961/2
class RandomNoise(object):
    def __init__(self, p, mean=0, std=1):
        self.p = p
        self.mean = mean
        self.std = std

    def __call__(self, img):
        if random.random() <= self.p:
            return img.clone().normal_(self.mean, self.std)
        return img


def canny_edge_detector(image):
    """
    Returns a binary image with same size of source image which each pixel determines belonging to an edge or not.

    :param image: PIL image
    :return: Binary numpy array
    """
    image = np.array(image)
    image = color.rgb2grey(image)
    edges = feature.canny(image, sigma=1)
    return edges * 1


def get_image_by_name(img_dir, name):
    """
    gets a image by a name gathered from file list csv file

    :param img_dir: Directory to image files as a uncompressed tar archive
    :param name: name of targeted image
    :return: a PIL image
    """

    with tarfile.open(img_dir) as tf:
        tarinfo = tf.getmember(name)
        image = tf.extractfile(tarinfo)
        image = image.read()
        image = Image.open(io.BytesIO(image))
    return image


# z = get_image_by_name('data/data.tar', 'Places365_val_00000002.jpg')