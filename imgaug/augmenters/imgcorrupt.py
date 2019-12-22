"""
Augmenters that wrap methods from ``bethgelab.imagecorruptions`` package.

See https://github.com/bethgelab/imagecorruptions for the package.

List of augmenters:

    * ImgcorruptGaussianNoise
    * ImgcorruptShotNoise
    * ImgcorruptImpulseNoise
    * ImgcorruptSpeckleNoise
    * ImgcorruptGaussianBlur
    * ImgcorruptGlassBlur
    * ImgcorruptDefocusBlur
    * ImgcorruptMotionBlur
    * ImgcorruptZoomBlur
    * ImgcorruptFog
    * ImgcorruptFrost
    * ImgcorruptSnow
    * ImgcorruptSpatter
    * ImgcorruptContrast
    * ImgcorruptBrightness
    * ImgcorruptSaturate
    * ImgcorruptJpegCompression
    * ImgcorruptPixelate
    * ImgcorruptElasticTransform

.. note::

    The functions provided here have identical outputs to the ones in
    ``imagecorruptions`` when called using the ``corrupt()`` function of
    that package. E.g. the outputs are always ``uint8`` and not
    ``float32`` or ``float64``.

Example usage::

    >>> import imgaug as ia
    >>> import imgaug.augmenters as iaa
    >>> import numpy as np
    >>> image = np.zeros((64, 64, 3), dtype=np.uint8)
    >>> names, funcs = iaa.get_imgcorrupt_subset("validation")
    >>> for name, func in zip(names, funcs):
    >>>     image_aug = func(image, severity=5, seed=1)
    >>>     image_aug = ia.draw_text(image, x=20, y=20, text=name)
    >>>     ia.imshow(image_aug)

"""
from __future__ import print_function, division, absolute_import

import warnings

import numpy as np

from .. import dtypes as iadt
from .. import random as iarandom
from .. import parameters as iap
from . import meta

# TODO add optional dependency

_MISSING_PACKAGE_ERROR_MSG = (
    "Could not import package `imagecorruptions`. This is an optional "
    "dependency of imgaug and must be installed manually in order "
    "to use augmenters from `imgaug.augmenters.imgcorrupt`. "
    "Use e.g. `pip install imagecorruptions` to install it. See also "
    "https://github.com/bethgelab/imagecorruptions for the repository "
    "of the package."
)


def _create_docstring(fname, image_arg_name="x"):
    return """Apply :func:`imagecorruptions.corruptions.%s`.

    dtype support::

        * ``uint8``: yes; indirectly tested (1)
        * ``uint16``: no
        * ``uint32``: no
        * ``uint64``: no
        * ``int8``: no
        * ``int16``: no
        * ``int32``: no
        * ``int64``: no
        * ``float16``: no
        * ``float32``: no
        * ``float64``: no
        * ``float128``: no
        * ``bool``: no
        
        - (1) Tested by comparison with function in ``imagecorruptions``
              package.

    Parameters
    ----------
    %s : ndarray
        Image array.
        Expected to have shape ``(H,W)``, ``(H,W,1)`` or ``(H,W,3)`` with
        dtype ``uint8`` and a minimum height/width of ``32``.

    severity : int, optional
        Strength of the corruption, with valid values being
        ``1 <= severity <= 5``.

    seed : None or int, optional
        Seed for the random number generation to use.

    Returns
    -------
    ndarray
        Corrupted image.

    """ % (fname, image_arg_name)


class _ImgcorruptAugmenterBase(meta.Augmenter):
    def __init__(self, func, severity=1,
                 name=None, deterministic=False, random_state=None):
        super(_ImgcorruptAugmenterBase, self).__init__(
            name=name, deterministic=deterministic, random_state=random_state)

        self.func = func
        self.severity = iap.handle_discrete_param(
            severity, "severity", value_range=(1, 5), tuple_to_uniform=True,
            list_to_choice=True, allow_floats=False)

    def _augment_batch(self, batch, random_state, parents, hooks):
        if batch.images is None:
            return batch

        severities, seeds = self._draw_samples(len(batch.images),
                                               random_state=random_state)

        for image, severity, seed in zip(batch.images, severities, seeds):
            image[...] = self.func(image, severity=severity, seed=seed)

        return batch

    def _draw_samples(self, nb_rows, random_state):
        severities = self.severity.draw_samples((nb_rows,),
                                                random_state=random_state)
        seeds = random_state.generate_seeds_(nb_rows)

        return severities, seeds

    def get_parameters(self):
        """See :func:`imgaug.augmenters.meta.Augmenter.get_parameters`."""
        return [self.severity]


def _create_augmenter(class_name, func_name):
    func = globals()["apply_imgcorrupt_%s" % (func_name,)]

    def __init__(self, severity=1, name=None, deterministic=False,
                 random_state=None):
        super(self.__class__, self).__init__(
            func, severity, name=name, deterministic=deterministic,
            random_state=random_state)

    augmenter_class = type(class_name,
                           (_ImgcorruptAugmenterBase,),
                           {"__init__": __init__})

    augmenter_class.__doc__ = """
    Wrapper around function :func:`imagecorruption.%s`.
    
    dtype support::
    
        See :func:`imgaug.augmenters.imgcorrupt.apply_imgcorrupt_%s`.
    
    Parameters
    ----------
    severity : int, optional
        Strength of the corruption, with valid values being
        ``1 <= severity <= 5``.
    
    name : None or str, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    deterministic : bool, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.bit_generator.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.
    
    Examples
    --------
    >>> import imgaug.augmenters as iaa
    >>> aug = iaa.%s(severity=2)
    
    Create an augmenter around :func:`imagecorruption.%s`. Apply it to
    images using e.g. ``aug(images=[image1, image2, ...])``.
    
    """ % (func_name, func_name, class_name, func_name)

    return augmenter_class


def _clipped_zoom_no_scipy_warning(img, zoom_factor):
    from scipy.ndimage import zoom as scizoom

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", ".*output shape of zoom.*")

        # clipping along the width dimension:
        ch0 = int(np.ceil(img.shape[0] / float(zoom_factor)))
        top0 = (img.shape[0] - ch0) // 2

        # clipping along the height dimension:
        ch1 = int(np.ceil(img.shape[1] / float(zoom_factor)))
        top1 = (img.shape[1] - ch1) // 2

        img = scizoom(img[top0:top0 + ch0, top1:top1 + ch1],
                      (zoom_factor, zoom_factor, 1), order=1)

        return img


def _call_imgcorrupt_func(fname, seed, convert_to_pil, *args, **kwargs):
    # import imagecorruptions, note that it is an optional dependency
    try:
        # imagecorruptions sets its own warnings filter rule via
        # warnings.simplefilter(). That rule is the in effect for the whole
        # program and not just the module. So to prevent that here
        # we use catch_warnings(), which uintuitively does not by default
        # catch warnings but saves and restores the warnings filter settings.
        with warnings.catch_warnings():
            import imagecorruptions.corruptions as corruptions
    except ImportError:
        raise ImportError(_MISSING_PACKAGE_ERROR_MSG)

    # Monkeypatch clip_zoom() as that causes warnings in some scipy versions,
    # and the implementation here suppresses these warnings. They suppress
    # all UserWarnings on a module level instead, which seems very exhaustive.
    corruptions.clipped_zoom = _clipped_zoom_no_scipy_warning

    image = args[0]

    iadt.gate_dtypes(
        image,
        allowed=["uint8"],
        disallowed=["bool",
                    "uint16", "uint32", "uint64", "uint128", "uint256",
                    "int8", "int16", "int32", "int64", "int128", "int256",
                    "float16", "float32", "float64", "float96", "float128",
                    "float256"],
        augmenter=None)

    input_shape = image.shape

    height, width = input_shape[0:2]
    assert height >= 32 and width >= 32, (
        "Expected the provided image to have a width and height of at least "
        "32 pixels, as that is the lower limit that the wrapped "
        "imagecorruptions functions use. Got shape %s." % (image.shape,))

    ndim = image.ndim
    assert ndim == 2 or (ndim == 3 and (image.shape[2] in [1, 3])), (
        "Expected input image to have shape (height, width) or "
        "(height, width, 1) or (height, width, 3). Got shape %s." % (
            image.shape,))

    if ndim == 2:
        image = image[..., np.newaxis]
    if image.shape[-1] == 1:
        image = np.tile(image, (1, 1, 3))

    if convert_to_pil:
        import PIL.Image
        image = PIL.Image.fromarray(image)

    with iarandom.temporary_numpy_seed(seed):
        image_aug = getattr(corruptions, fname)(image, *args[1:], **kwargs)

    if convert_to_pil:
        image_aug = np.asarray(image_aug)

    if ndim == 2:
        image_aug = image_aug[:, :, 0]
    elif input_shape[-1] == 1:
        image_aug = image_aug[:, :, 0:1]

    # this cast is done at the end of imagecorruptions.__init__.corrupt()
    image_aug = np.uint8(image_aug)

    return image_aug


def get_imgcorrupt_subset(subset="common"):
    """Get a named subset of image corruption functions.

    Parameters
    ----------
    subset : {'common', 'validation', 'all'}, optional.
        Name of the subset of image corruption functions.

    Returns
    -------
    list of str
        Names of the corruption methods, e.g. "gaussian_noise".

    list of callable
        Function corresponding to the name. Is one of the
        ``apply_imgcorrupt_*()`` functions in this module. Apply e.g.
        via ``func(image, severity=2, seed=123)``.

    """
    # import imagecorruptions, note that it is an optional dependency
    try:
        # imagecorruptions sets its own warnings filter rule via
        # warnings.simplefilter(). That rule is the in effect for the whole
        # program and not just the module. So to prevent that here
        # we use catch_warnings(), which uintuitively does not by default
        # catch warnings but saves and restores the warnings filter settings.
        with warnings.catch_warnings():
            import imagecorruptions
    except ImportError:
        raise ImportError(_MISSING_PACKAGE_ERROR_MSG)

    cnames = imagecorruptions.get_corruption_names(subset)
    funcs = [globals()["apply_imgcorrupt_%s" % (cname,)] for cname in cnames]

    return cnames, funcs


def apply_imgcorrupt_gaussian_noise(x, severity=1, seed=None):
    return _call_imgcorrupt_func("gaussian_noise", seed, False, x, severity)

apply_imgcorrupt_gaussian_noise.__doc__ = _create_docstring("gaussian_noise")


def apply_imgcorrupt_shot_noise(x, severity=1, seed=None):
    return _call_imgcorrupt_func("shot_noise", seed, False, x, severity)

apply_imgcorrupt_shot_noise.__doc__ = _create_docstring("shot_noise")


def apply_imgcorrupt_impulse_noise(x, severity=1, seed=None):
    return _call_imgcorrupt_func("impulse_noise", seed, False, x, severity)

apply_imgcorrupt_impulse_noise.__doc__ = _create_docstring("impulse_noise")


def apply_imgcorrupt_speckle_noise(x, severity=1, seed=None):
    return _call_imgcorrupt_func("speckle_noise", seed, False, x, severity)

apply_imgcorrupt_speckle_noise.__doc__ = _create_docstring("speckle_noise")


def apply_imgcorrupt_gaussian_blur(x, severity=1, seed=None):
    return _call_imgcorrupt_func("gaussian_blur", seed, False, x, severity)

apply_imgcorrupt_gaussian_blur.__doc__ = _create_docstring("gaussian_blur")


def apply_imgcorrupt_glass_blur(x, severity=1, seed=None):
    return _call_imgcorrupt_func("glass_blur", seed, False, x, severity)

apply_imgcorrupt_glass_blur.__doc__ = _create_docstring("glass_blur")


def apply_imgcorrupt_defocus_blur(x, severity=1, seed=None):
    return _call_imgcorrupt_func("defocus_blur", seed, False, x, severity)

apply_imgcorrupt_defocus_blur.__doc__ = _create_docstring("defocus_blur")


def apply_imgcorrupt_motion_blur(x, severity=1, seed=None):
    return _call_imgcorrupt_func("motion_blur", seed, False, x, severity)

apply_imgcorrupt_motion_blur.__doc__ = _create_docstring("motion_blur")


def apply_imgcorrupt_zoom_blur(x, severity=1, seed=None):
    return _call_imgcorrupt_func("zoom_blur", seed, False, x, severity)

apply_imgcorrupt_zoom_blur.__doc__ = _create_docstring("zoom_blur")


def apply_imgcorrupt_fog(x, severity=1, seed=None):
    return _call_imgcorrupt_func("fog", seed, False, x, severity)

apply_imgcorrupt_fog.__doc__ = _create_docstring("fog")


def apply_imgcorrupt_frost(x, severity=1, seed=None):
    return _call_imgcorrupt_func("frost", seed, False, x, severity)

apply_imgcorrupt_frost.__doc__ = _create_docstring("frost")


def apply_imgcorrupt_snow(x, severity=1, seed=None):
    return _call_imgcorrupt_func("snow", seed, False, x, severity)

apply_imgcorrupt_snow.__doc__ = _create_docstring("snow")


def apply_imgcorrupt_spatter(x, severity=1, seed=None):
    return _call_imgcorrupt_func("spatter", seed, True, x, severity)

apply_imgcorrupt_spatter.__doc__ = _create_docstring("spatter")


def apply_imgcorrupt_contrast(x, severity=1, seed=None):
    return _call_imgcorrupt_func("contrast", seed, False, x, severity)

apply_imgcorrupt_contrast.__doc__ = _create_docstring("contrast")


def apply_imgcorrupt_brightness(x, severity=1, seed=None):
    return _call_imgcorrupt_func("brightness", seed, False, x, severity)

apply_imgcorrupt_brightness.__doc__ = _create_docstring("brightness")


def apply_imgcorrupt_saturate(x, severity=1, seed=None):
    return _call_imgcorrupt_func("saturate", seed, False, x, severity)

apply_imgcorrupt_saturate.__doc__ = _create_docstring("saturate")


def apply_imgcorrupt_jpeg_compression(x, severity=1, seed=None):
    return _call_imgcorrupt_func("jpeg_compression", seed, True, x, severity)

apply_imgcorrupt_jpeg_compression.__doc__ = _create_docstring(
    "jpeg_compression")


def apply_imgcorrupt_pixelate(x, severity=1, seed=None):
    return _call_imgcorrupt_func("pixelate", seed, True, x, severity)

apply_imgcorrupt_pixelate.__doc__ = _create_docstring("pixelate")


def apply_imgcorrupt_elastic_transform(image, severity=1, seed=None):
    return _call_imgcorrupt_func("elastic_transform", seed, False, image,
                                 severity)

apply_imgcorrupt_elastic_transform.__doc__ = _create_docstring(
    "elastic_transform")


ImgcorruptGaussianNoise = _create_augmenter("ImgcorruptGaussianNoise",
                                            "gaussian_noise")
ImgcorruptShotNoise = _create_augmenter("ImgcorruptShotNoise",
                                        "shot_noise")
ImgcorruptImpulseNoise = _create_augmenter("ImgcorruptImpulseNoise",
                                           "impulse_noise")
ImgcorruptSpeckleNoise = _create_augmenter("ImgcorruptSpeckleNoise",
                                           "speckle_noise")
ImgcorruptGaussianBlur = _create_augmenter("ImgcorruptGaussianBlur",
                                           "gaussian_blur")
ImgcorruptGlassBlur = _create_augmenter("ImgcorruptGlassBlur",
                                        "glass_blur")
ImgcorruptDefocusBlur = _create_augmenter("ImgcorruptDefocusBlur",
                                          "defocus_blur")
ImgcorruptMotionBlur = _create_augmenter("ImgcorruptMotionBlur",
                                         "motion_blur")
ImgcorruptZoomBlur = _create_augmenter("ImgcorruptZoomBlur",
                                       "zoom_blur")
ImgcorruptFog = _create_augmenter("ImgcorruptFog",
                                  "fog")
ImgcorruptFrost = _create_augmenter("ImgcorruptFrost",
                                    "frost")
ImgcorruptSnow = _create_augmenter("ImgcorruptSnow",
                                   "snow")
ImgcorruptSpatter = _create_augmenter("ImgcorruptSpatter",
                                      "spatter")
ImgcorruptContrast = _create_augmenter("ImgcorruptContrast",
                                       "contrast")
ImgcorruptBrightness = _create_augmenter("ImgcorruptBrightness",
                                         "brightness")
ImgcorruptSaturate = _create_augmenter("ImgcorruptSaturate",
                                       "saturate")
ImgcorruptJpegCompression = _create_augmenter("ImgcorruptJpegCompression",
                                              "jpeg_compression")
ImgcorruptPixelate = _create_augmenter("ImgcorruptPixelate",
                                       "pixelate")
ImgcorruptElasticTransform = _create_augmenter("ImgcorruptElasticTransform",
                                               "elastic_transform")
