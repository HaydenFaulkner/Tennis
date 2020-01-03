import mxnet as mx

from gluoncv.data.transforms.experimental.image import random_color_distort
from gluoncv.data.transforms.image import random_pca_lighting, random_flip


class TwoStreamTransform(object):

    def __init__(self, color_dist=True,
                 mean=(0.485, 0.456, 0.406, 0.863, 0.871, 0.883),
                 std=(0.229, 0.224, 0.225, 0.098, 0.087, 0.095)):

        self._mean = mean
        self._std = std
        self._cd = color_dist

    def __call__(self, img):
        """Apply transform to image."""

        img, _ = random_flip(img, 0.5, 0)

        if self._cd:
            img[:, :, :3] = random_color_distort(img[:, :, :3])
            # img[:, :, :3] = random_pca_lighting(img[:, :, :3], 0.1)
        img = mx.nd.image.to_tensor(img)
        img[:3, :, :] = mx.nd.image.normalize(img[:3, :, :], mean=self._mean[:3], std=self._std[:3])
        if len(self._mean) > 3 and len(self._std) > 3:
            img[3:, :, :] = mx.nd.image.normalize(img[3:, :, :], mean=self._mean[3:], std=self._std[3:])

        return img


