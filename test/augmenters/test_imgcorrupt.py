from __future__ import print_function, division, absolute_import

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
import functools

import matplotlib
matplotlib.use('Agg')  # fix execution of tests involving matplotlib on travis
import numpy as np
import imagecorruptions
import imagecorruptions.corruptions as corruptions
from imagecorruptions import corrupt
import PIL.Image

import imgaug as ia
from imgaug import augmenters as iaa
from imgaug import random as iarandom


class Test_get_imgcorrupt_subset(unittest.TestCase):
    def test_by_comparison_with_imagecorruptions(self):
        subset_names = ["common", "validation", "all"]
        for subset in subset_names:
            with self.subTest(subset=subset):
                func_names, funcs = iaa.get_imgcorrupt_subset(subset)
                func_names_exp = imagecorruptions.get_corruption_names(subset)

                assert func_names == func_names_exp
                for func_name, func in zip(func_names, funcs):
                    assert getattr(
                        iaa, "apply_imgcorrupt_%s" % (func_name,)
                    ) is func

    def test_subset_functions(self):
        subset_names = ["common", "validation", "all"]
        for subset in subset_names:
            func_names, funcs = iaa.get_imgcorrupt_subset(subset)
            image = np.mod(
                np.arange(32*32*3), 256
            ).reshape((32, 32, 3)).astype(np.uint8)

            for func_name, func in zip(func_names, funcs):
                with self.subTest(subset=subset, name=func_name):
                    # don't verify here whether e.g. only seed 2 produces
                    # different results from seed 1, because some methods
                    # are only dependent on the severity
                    image_aug1 = func(image, severity=5, seed=1)
                    image_aug2 = func(image, severity=5, seed=1)
                    image_aug3 = func(image, severity=1, seed=2)
                    assert not np.array_equal(image, image_aug1)
                    assert not np.array_equal(image, image_aug2)
                    assert not np.array_equal(image_aug2, image_aug3)
                    assert np.array_equal(image_aug1, image_aug2)


class _CompareFuncWithImageCorruptions(unittest.TestCase):
    def _test_by_comparison_with_imagecorruptions(
            self,
            fname,
            shapes=((64, 64), (64, 64, 1), (64, 64, 3)),
            dtypes=("uint8",),
            severities=(1, 2, 3, 4, 5),
            seeds=(1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
            convert_to_pil=False):
        for shape in shapes:
            for dtype in dtypes:
                for severity in severities:
                    for seed in seeds:
                        with self.subTest(shape=shape, severity=severity,
                                          seed=seed):
                            image_imgaug = self.create_image_imgaug(
                                shape, dtype, 1000 + seed)
                            image_imgcor = np.copy(image_imgaug)

                            self._run_single_comparison_test(
                                fname, image_imgaug, image_imgcor, severity,
                                seed)

    @classmethod
    def create_image_imgaug(cls, shape, dtype, seed, tile=None):
        rng = iarandom.RNG(1000 + seed)

        if dtype.startswith("uint"):
            image = rng.integers(0, 256, size=shape, dtype=dtype)
        else:
            assert dtype.startswith("float")
            image = rng.uniform(0.0, 1.0, size=shape)
            image = image.astype(dtype)

        if tile is not None:
            image = np.tile(image, tile)

        return image

    @classmethod
    def _run_single_comparison_test(cls, fname, image_imgaug, image_imgcor,
                                    severity, seed):
        image_imgaug_sum = np.sum(image_imgaug)
        image_imgcor_sum = np.sum(image_imgcor)

        image_aug, image_aug_exp = cls._generate_augmented_images(
            fname, image_imgaug, image_imgcor, severity, seed)

        # assert that the original image is unchanged,
        # i.e. it was not augmented in-place
        assert np.isclose(np.sum(image_imgcor), image_imgcor_sum, rtol=0,
                          atol=1e-4)
        assert np.isclose(np.sum(image_imgaug), image_imgaug_sum, rtol=0,
                          atol=1e-4)

        # assert that the functions returned numpy arrays and not PIL images
        assert ia.is_np_array(image_aug_exp)
        assert ia.is_np_array(image_aug)

        assert image_aug.shape == image_imgaug.shape
        assert image_aug.dtype.name == image_aug_exp.dtype.name

        atol = 1e-4  # set this to 0.5+1e-4 if output is converted to uint8
        assert np.allclose(image_aug, image_aug_exp, rtol=0, atol=atol)

    @classmethod
    def _generate_augmented_images(cls, fname, image_imgaug, image_imgcor,
                                   severity, seed):
        func_imgaug = getattr(
            iaa,
            "apply_imgcorrupt_%s" % (fname,))
        func_imagecor = functools.partial(corrupt, corruption_name=fname)

        with iarandom.temporary_numpy_seed(seed):
            image_aug_exp = func_imagecor(image_imgcor, severity=severity)
            if not ia.is_np_array(image_aug_exp):
                image_aug_exp = np.asarray(image_aug_exp)
            if image_imgcor.ndim == 2:
                image_aug_exp = image_aug_exp[:, :, 0]
            elif image_imgcor.shape[-1] == 1:
                image_aug_exp = image_aug_exp[:, :, 0:1]

        image_aug = func_imgaug(image_imgaug, severity=severity,
                                seed=seed)

        return image_aug, image_aug_exp


class Test_apply_functions(_CompareFuncWithImageCorruptions):
    def test_apply_imgcorrupt_gaussian_noise(self):
        self._test_by_comparison_with_imagecorruptions("gaussian_noise")

    def test_apply_imgcorrupt_shot_noise(self):
        self._test_by_comparison_with_imagecorruptions("shot_noise")

    def test_apply_imgcorrupt_impulse_noise(self):
        self._test_by_comparison_with_imagecorruptions("impulse_noise")

    def test_apply_imgcorrupt_speckle_noise(self):
        self._test_by_comparison_with_imagecorruptions("speckle_noise")

    def test_apply_imgcorrupt_gaussian_blur(self):
        self._test_by_comparison_with_imagecorruptions("gaussian_blur")

    def test_apply_imgcorrupt_glass_blur(self):
        # glass_blur() is extremely slow, so we run only a reduced set
        # of tests here
        self._test_by_comparison_with_imagecorruptions(
            "glass_blur",
            shapes=[(32, 32), (32, 32, 1), (32, 32, 3)],
            severities=[3],
            seeds=[1])

    def test_apply_imgcorrupt_defocus_blur(self):
        self._test_by_comparison_with_imagecorruptions(
            "defocus_blur")

    def test_apply_imgcorrupt_motion_blur(self):
        self._test_by_comparison_with_imagecorruptions(
            "motion_blur")

    def test_apply_imgcorrupt_zoom_blur(self):
        self._test_by_comparison_with_imagecorruptions(
            "zoom_blur")

    def test_apply_imgcorrupt_fog(self):
        self._test_by_comparison_with_imagecorruptions(
            "fog")

    def test_apply_imgcorrupt_frost(self):
        self._test_by_comparison_with_imagecorruptions(
            "frost",
            severities=[1, 5],
            seeds=[1, 5, 10])

    def test_apply_imgcorrupt_snow(self):
        self._test_by_comparison_with_imagecorruptions(
            "snow")

    def test_apply_imgcorrupt_spatter(self):
        self._test_by_comparison_with_imagecorruptions(
            "spatter")

    def test_apply_imgcorrupt_contrast(self):
        self._test_by_comparison_with_imagecorruptions("contrast")

    def test_apply_imgcorrupt_brightness(self):
        self._test_by_comparison_with_imagecorruptions("brightness")

    def test_apply_imgcorrupt_saturate(self):
        self._test_by_comparison_with_imagecorruptions(
            "saturate")

    def test_apply_imgcorrupt_jpeg_compression(self):
        self._test_by_comparison_with_imagecorruptions(
            "jpeg_compression")

    def test_apply_imgcorrupt_pixelate(self):
        self._test_by_comparison_with_imagecorruptions(
            "pixelate")

    def test_apply_imgcorrupt_elastic_transform(self):
        self._test_by_comparison_with_imagecorruptions(
            "elastic_transform")
