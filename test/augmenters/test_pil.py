from __future__ import print_function, division, absolute_import

import itertools
import sys
# unittest only added in 3.4 self.subTest()
if sys.version_info[0] < 3 or sys.version_info[1] < 4:
    import unittest2 as unittest
else:
    import unittest
# unittest.mock is not available in 2.7 (though unittest2 might contain it?)
try:
    import unittest.mock as mock
except ImportError:
    import mock
import warnings

import matplotlib
matplotlib.use('Agg')  # fix execution of tests involving matplotlib on travis
import numpy as np
import six.moves as sm
import skimage
import skimage.data
import cv2
import PIL.Image
import PIL.ImageOps

import imgaug as ia
from imgaug import augmenters as iaa
from imgaug import parameters as iap
from imgaug import dtypes as iadt
from imgaug import random as iarandom
from imgaug.augmenters import contrast as contrast_lib
from imgaug.augmentables import batches as iabatches
from imgaug.testutils import (ArgCopyingMagicMock, keypoints_equal, reseed,
                              runtest_pickleable_uint8_img)


class Test_pil_equalize(unittest.TestCase):
    def test_by_comparison_with_pil(self):
        shapes = [
            (1, 1),
            (2, 1),
            (1, 2),
            (2, 2),
            (5, 5),
            (10, 5),
            (5, 10),
            (10, 10),
            (20, 20),
            (100, 100),
            (100, 200),
            (200, 100),
            (200, 200)
        ]
        shapes = shapes + [shape + (3,) for shape in shapes]

        rng = iarandom.RNG(0)
        images = [rng.integers(0, 255, size=shape).astype(np.uint8)
                  for shape in shapes]
        images = images + [
            np.full((10, 10), 0, dtype=np.uint8),
            np.full((10, 10), 128, dtype=np.uint8),
            np.full((10, 10), 255, dtype=np.uint8)
        ]

        for i, image in enumerate(images):
            mask_vals = [False, True] if image.size >= (100*100) else [False]
            for use_mask in mask_vals:
                with self.subTest(image_idx=i, shape=image.shape,
                                  use_mask=use_mask):
                    mask_np = None
                    mask_pil = None
                    if use_mask:
                        mask_np = np.zeros(image.shape[0:2], dtype=np.uint8)
                        mask_np[25:75, 25:75] = 1
                        mask_pil = PIL.Image.fromarray(mask_np).convert("L")

                    image_iaa = iaa.pil_equalize(image, mask=mask_np)
                    image_pil = np.asarray(
                        PIL.ImageOps.equalize(
                            PIL.Image.fromarray(image),
                            mask=mask_pil
                        )
                    )

                    assert np.array_equal(image_iaa, image_pil)

    def test_unusual_channel_numbers(self):
        nb_channels_lst = [1, 2, 4, 5, 512, 513]
        for nb_channels in nb_channels_lst:
            for size in [20, 100]:
                with self.subTest(nb_channels=nb_channels,
                                  size=size):
                    shape = (size, size, nb_channels)
                    image = iarandom.RNG(0).integers(50, 150, size=shape)
                    image = image.astype(np.uint8)

                    image_aug = iaa.pil_equalize(image)

                    if size > 1:
                        channelwise_sums = np.sum(image_aug, axis=(0, 1))
                        assert np.all(channelwise_sums > 0)
                    assert np.min(image_aug) < 50
                    assert np.max(image_aug) > 150

    def test_zero_sized_axes(self):
        shapes = [
            (0, 0),
            (0, 1),
            (1, 0),
            (0, 1, 0),
            (1, 0, 0),
            (0, 1, 1),
            (1, 0, 1)
        ]

        for shape in shapes:
            with self.subTest(shape=shape):
                image = np.zeros(shape, dtype=np.uint8)

                image_aug = iaa.pil_equalize(image)

                assert image_aug.dtype.name == "uint8"
                assert image_aug.shape == shape


class Test_pil_autocontrast(unittest.TestCase):
    def test_by_comparison_with_pil(self):
        rng = iarandom.RNG(0)
        shapes = [
            (1, 1),
            (10, 10),
            (1, 1, 3),
            (1, 2, 3),
            (2, 1, 3),
            (2, 2, 3),
            (5, 3, 3),
            (10, 5, 3),
            (5, 10, 3),
            (10, 10, 3),
            (20, 10, 3),
            (20, 40, 3),
            (50, 60, 3),
            (100, 100, 3),
            (200, 100, 3)
        ]
        images = [
            rng.integers(0, 255, size=shape).astype(np.uint8)
            for shape in shapes
        ]
        images = (
            images
            + [
                np.full((1, 1, 3), 0, dtype=np.uint8),
                np.full((1, 1, 3), 255, dtype=np.uint8),
                np.full((20, 20, 3), 0, dtype=np.uint8),
                np.full((20, 20, 3), 255, dtype=np.uint8)
            ]
        )

        cutoffs = [0, 1, 2, 10, 50, 90, 99, 100]
        ignores = [None, 0, 1, 100, 255, [0, 1], [5, 10, 50], [99, 100]]

        for cutoff in cutoffs:
            for ignore in ignores:
                for i, image in enumerate(images):
                    with self.subTest(cutoff=cutoff, ignore=ignore,
                                      image_idx=i, image_shape=image.shape):
                        result_pil = np.asarray(
                            PIL.ImageOps.autocontrast(
                                PIL.Image.fromarray(image),
                                cutoff=cutoff,
                                ignore=ignore
                            )
                        )
                        result_iaa = iaa.pil_autocontrast(image,
                                                          cutoff=cutoff,
                                                          ignore=ignore)
                        assert np.array_equal(result_pil, result_iaa)

    def test_unusual_channel_numbers(self):
        nb_channels_lst = [1, 2, 4, 5, 512, 513]
        for nb_channels in nb_channels_lst:
            for size in [20]:
                for cutoff in [0, 1, 10]:
                    with self.subTest(nb_channels=nb_channels,
                                      size=size,
                                      cutoff=cutoff):
                        shape = (size, size, nb_channels)
                        image = iarandom.RNG(0).integers(50, 150, size=shape)
                        image = image.astype(np.uint8)

                        image_aug = iaa.pil_autocontrast(image, cutoff=cutoff)

                        if size > 1:
                            channelwise_sums = np.sum(image_aug, axis=(0, 1))
                            assert np.all(channelwise_sums > 0)
                        assert np.min(image_aug) < 50
                        assert np.max(image_aug) > 150

    def test_zero_sized_axes(self):
        shapes = [
            (0, 0),
            (0, 1),
            (1, 0),
            (0, 1, 0),
            (1, 0, 0),
            (0, 1, 1),
            (1, 0, 1)
        ]

        for shape in shapes:
            for cutoff in [0, 1, 10]:
                for ignore in [None, 0, 1, [0, 1, 10]]:
                    with self.subTest(shape=shape, cutoff=cutoff,
                                      ignore=ignore):
                        image = np.zeros(shape, dtype=np.uint8)

                        image_aug = iaa.pil_autocontrast(image, cutoff=cutoff,
                                                         ignore=ignore)

                        assert image_aug.dtype.name == "uint8"
                        assert image_aug.shape == shape


class TestPILSolarize(unittest.TestCase):
    def test_returns_solarize(self):
        aug = iaa.PILSolarize(0.05)
        assert isinstance(aug, iaa.Solarize)


class TestPILPosterize(unittest.TestCase):
    def test_returns_posterize(self):
        aug = iaa.PILPosterize()
        assert isinstance(aug, iaa.Posterize)


class TestEqualize(unittest.TestCase):
    def setUp(self):
        reseed()

    @mock.patch("imgaug.augmenters.pil.pil_equalize_")
    def test_mocked(self, mock_eq):
        image = np.arange(1*1*3).astype(np.uint8).reshape((1, 1, 3))
        mock_eq.return_value = np.copy(image)
        aug = iaa.PILEqualize()

        _image_aug = aug(image=image)

        assert mock_eq.call_count == 1
        assert np.array_equal(mock_eq.call_args_list[0][0][0], image)

    def test_integrationtest(self):
        rng = iarandom.RNG(0)
        for size in [20, 100]:
            shape = (size, size, 3)
            image = rng.integers(50, 150, size=shape)
            image = image.astype(np.uint8)
            aug = iaa.PILEqualize()

            image_aug = aug(image=image)

            if size > 1:
                channelwise_sums = np.sum(image_aug, axis=(0, 1))
                assert np.all(channelwise_sums > 0)
            assert np.min(image_aug) < 50
            assert np.max(image_aug) > 150


class TestPILAutocontrast(unittest.TestCase):
    def setUp(self):
        reseed()

    @mock.patch("imgaug.augmenters.pil.pil_autocontrast")
    def test_mocked(self, mock_auto):
        image = np.mod(np.arange(10*10*3), 255)
        image = image.reshape((10, 10, 3)).astype(np.uint8)
        mock_auto.return_value = image
        aug = iaa.PILAutocontrast(15)

        _image_aug = aug(image=image)

        assert np.array_equal(mock_auto.call_args_list[0][0][0], image)
        assert mock_auto.call_args_list[0][0][1] == 15

    @mock.patch("imgaug.augmenters.pil.pil_autocontrast")
    def test_per_channel(self, mock_auto):
        image = np.mod(np.arange(10*10*1), 255)
        image = image.reshape((10, 10, 1)).astype(np.uint8)
        image = np.tile(image, (1, 1, 100))
        mock_auto.return_value = image[..., 0]
        aug = iaa.PILAutocontrast((0, 30), per_channel=True)

        _image_aug = aug(image=image)

        assert mock_auto.call_count == 100
        cutoffs = []
        for i in np.arange(100):
            assert np.array_equal(mock_auto.call_args_list[i][0][0],
                                  image[..., i])
            cutoffs.append(mock_auto.call_args_list[i][0][1])
        assert len(set(cutoffs)) > 10

    def test_integrationtest(self):
        image = iarandom.RNG(0).integers(50, 150, size=(100, 100, 3))
        image = image.astype(np.uint8)
        aug = iaa.PILAutocontrast(10)

        image_aug = aug(image=image)

        assert np.min(image_aug) < 50
        assert np.max(image_aug) > 150

    def test_integrationtest_per_channel(self):
        image = iarandom.RNG(0).integers(50, 150, size=(100, 100, 50))
        image = image.astype(np.uint8)
        aug = iaa.PILAutocontrast(10, per_channel=True)

        image_aug = aug(image=image)

        assert np.min(image_aug) < 50
        assert np.max(image_aug) > 150
