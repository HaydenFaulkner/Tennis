import mxnet as mx


class TwoStreamNormalize(object):
    def __init__(self,
                 mean=(0.485, 0.456, 0.406, 0.863, 0.871, 0.883),
                 std=(0.229, 0.224, 0.225, 0.098, 0.087, 0.095)):

        self._mean = mean
        self._std = std

    def __call__(self, img):
        """Apply transform to image."""

        img = mx.nd.image.to_tensor(img)
        img[:3, :, :] = mx.nd.image.normalize(img[:3, :, :], mean=self._mean[:3], std=self._std[:3])
        img[3:, :, :] = mx.nd.image.normalize(img[3:, :, :], mean=self._mean[3:], std=self._std[3:])

        return img
