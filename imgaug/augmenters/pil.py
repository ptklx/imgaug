"""
Augmenters that have identical outputs to well-known PIL functions.

Some of the augmenters in this module may also exist in other modules
under similar name. These other augmenters may currently have the same
outputs as the corresponding PIL functions, but that is not guaranteed
for the future. Use the augmenters in this module if identical outputs
to PIL are required.

The augmenters in this module may partially wrap PIL. They are usually
about as fast as PIL and in some cases faster.

List of augmenters:

    * PILSolarize
    * PILPosterize
    * PILEqualize
    * PILAutocontrast

"""
from __future__ import print_function, division, absolute_import

import re
import functools

import six.moves as sm
import numpy as np
import cv2
import PIL.Image
import PIL.ImageOps

import imgaug as ia
from . import meta
from . import arithmetic
from . import color
from . import contrast
from .. import parameters as iap


_EQUALIZE_USE_PIL_BELOW = 64*64  # H*W


def pil_equalize(image, mask=None):
    """Equalize the image histogram.

    See :func:`pil_equalize_` for details.

    This function is identical in inputs and outputs to
    :func:`PIL.ImageOps.equalize`.

    dtype support::

        See :func:`imgaug.augmenters.pil.pil_equalize_`.

    Parameters
    ----------
    image : ndarray
        ``uint8`` ``(H,W,[C])`` image to equalize.

    mask : None or ndarray, optional
        An optional mask. If given, only the pixels selected by the mask are
        included in the analysis.

    Returns
    -------
    ndarray
        Equalized image.

    """
    # internally used method works in-place by default and hence needs a copy
    size = image.size
    if size == 0:
        return np.copy(image)
    if size >= _EQUALIZE_USE_PIL_BELOW:
        image = np.copy(image)
    return pil_equalize_(image, mask)


def pil_equalize_(image, mask=None):
    """Equalize the image histogram in-place.

    This function applies a non-linear mapping to the input image, in order
    to create a uniform distribution of grayscale values in the output image.

    This function is identical in inputs and outputs to
    :func:`PIL.ImageOps.equalize`, except that it is allowed to modify the
    input image in-place.

    dtype support::

        * ``uint8``: yes; fully tested
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

    Parameters
    ----------
    image : ndarray
        ``uint8`` ``(H,W,[C])`` image to equalize.

    mask : None or ndarray, optional
        An optional mask. If given, only the pixels selected by the mask are
        included in the analysis.

    Returns
    -------
    ndarray
        Equalized image. *Might* have been modified in-place.

    """
    nb_channels = 1 if image.ndim == 2 else image.shape[-1]
    if nb_channels not in [1, 3]:
        result = [pil_equalize_(image[:, :, c])
                  for c in np.arange(nb_channels)]
        return np.stack(result, axis=-1)

    assert image.dtype.name == "uint8", (
        "Expected image of dtype uint8, got dtype %s." % (image.dtype.name,))
    if mask is not None:
        assert mask.ndim == 2, (
            "Expected 2-dimensional mask, got shape %s." % (mask.shape,))
        assert mask.dtype.name == "uint8", (
            "Expected mask of dtype uint8, got dtype %s." % (mask.dtype.name,))

    size = image.size
    if size == 0:
        return image
    if nb_channels == 3 and size < _EQUALIZE_USE_PIL_BELOW:
        return _pil_equalize_pil(image, mask)
    return _pil_equalize_no_pil_(image, mask)


# note that this is supposed to be a non-PIL reimplementation of PIL's
# equalize, which produces slightly different results from cv2.equalizeHist()
def _pil_equalize_no_pil_(image, mask=None):
    flags = image.flags
    if not flags["OWNDATA"]:
        image = np.copy(image)
    if not flags["C_CONTIGUOUS"]:
        image = np.ascontiguousarray(image)

    nb_channels = 1 if image.ndim == 2 else image.shape[-1]
    lut = np.empty((1, 256, nb_channels), dtype=np.int32)

    for c_idx in range(nb_channels):
        if image.ndim == 2:
            image_c = image[:, :, np.newaxis]
        else:
            image_c = image[:, :, c_idx:c_idx+1]
        histo = cv2.calcHist([image_c], [0], mask, [256], [0, 256])
        if len(histo.nonzero()[0]) <= 1:
            lut[0, :, c_idx] = np.arange(256).astype(np.int32)
            continue

        step = np.sum(histo[:-1]) // 255
        if not step:
            lut[0, :, c_idx] = np.arange(256).astype(np.int32)
            continue

        n = step // 2
        cumsum = np.cumsum(histo)
        lut[0, 0, c_idx] = n
        lut[0, 1:, c_idx] = n + cumsum[0:-1]
        lut[0, :, c_idx] //= int(step)
    lut = np.clip(lut, None, 255, out=lut).astype(np.uint8)
    image = cv2.LUT(image, lut, dst=image)
    if image.ndim == 2 and image.ndim == 3:
        return image[..., np.newaxis]
    return image


def _pil_equalize_pil(image, mask=None):
    if mask is not None:
        mask = PIL.Image.fromarray(mask).convert("L")
    return np.asarray(
        PIL.ImageOps.equalize(
            PIL.Image.fromarray(image),
            mask=mask
        )
    )


def pil_autocontrast(image, cutoff=0, ignore=None):
    """Maximize (normalize) image contrast.

    This function calculates a histogram of the input image, removes
    **cutoff** percent of the lightest and darkest pixels from the histogram,
    and remaps the image so that the darkest pixel becomes black (``0``), and
    the lightest becomes white (``255``).

    This function has identical inputs and outputs to
    :func:`PIL.ImageOps.autocontrast`. The speed almost identical.

    dtype support::

        * ``uint8``: yes; fully tested
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

    Parameters
    ----------
    image : ndarray
        The image for which to enhance the contrast.

    cutoff : number
        How many percent to cut off at the low and high end of the
        histogram. E.g. ``20`` will cut off the lowest and highest ``20%``
        of values. Expected value range is ``[0, 100]``.

    ignore : None or int or iterable of int
        Intensity values to ignore, i.e. to treat as background. If ``None``,
        no pixels will be ignored. Otherwise exactly the given intensity
        value(s) will be ignored.

    Returns
    -------
    ndarray
        Contrast-enhanced image.

    """
    assert image.dtype.name == "uint8", (
        "Can only apply autocontrast to uint8 images, got dtype %s." % (
            image.dtype.name,))

    if 0 in image.shape:
        return np.copy(image)

    standard_channels = (image.ndim == 2 or image.shape[2] == 3)

    if cutoff and standard_channels:
        return _pil_autocontrast_pil(image, cutoff, ignore)
    return _pil_autocontrast_no_pil(image, cutoff, ignore)


def _pil_autocontrast_pil(image, cutoff, ignore):
    return np.asarray(
        PIL.ImageOps.autocontrast(
            PIL.Image.fromarray(image),
            cutoff=cutoff, ignore=ignore
        )
    )


# This function is only faster than the corresponding PIL function if no
# cutoff is used.
# C901 is "<functionname> is too complex"
def _pil_autocontrast_no_pil(image, cutoff, ignore):  # noqa: C901
    # pylint: disable=invalid-name
    if ignore is not None and not ia.is_iterable(ignore):
        ignore = [ignore]

    result = np.empty_like(image)
    if result.ndim == 2:
        result = result[..., np.newaxis]
    nb_channels = image.shape[2] if image.ndim >= 3 else 1
    for c_idx in sm.xrange(nb_channels):
        # using [0] instead of [int(c_idx)] allows this to work with >4
        # channels
        if image.ndim == 2:
            image_c = image
        else:
            image_c = image[:, :, c_idx:c_idx+1]
        h = cv2.calcHist([image_c], [0], None, [256], [0, 256])
        if ignore is not None:
            h[ignore] = 0

        if cutoff:
            cs = np.cumsum(h)
            n = cs[-1]
            cut = n * cutoff // 100

            # remove cutoff% pixels from the low end
            lo_cut = cut - cs
            lo_cut_nz = np.nonzero(lo_cut <= 0.0)[0]
            if len(lo_cut_nz) == 0:
                lo = 255
            else:
                lo = lo_cut_nz[0]
            if lo > 0:
                h[:lo] = 0
            h[lo] = lo_cut[lo]

            # remove cutoff% samples from the hi end
            cs_rev = np.cumsum(h[::-1])
            hi_cut = cs_rev - cut
            hi_cut_nz = np.nonzero(hi_cut > 0.0)[0]
            if len(hi_cut_nz) == 0:
                hi = -1
            else:
                hi = 255 - hi_cut_nz[0]
            h[hi+1:] = 0
            if hi > -1:
                h[hi] = hi_cut[255-hi]

        # find lowest/highest samples after preprocessing
        for lo, lo_val in enumerate(h):
            if lo_val:
                break
        for hi in range(255, -1, -1):
            if h[hi]:
                break
        if hi <= lo:
            # don't bother
            lut = np.arange(256)
        else:
            scale = 255.0 / (hi - lo)
            offset = -lo * scale
            ix = np.arange(256).astype(np.float64) * scale + offset
            ix = np.clip(ix, 0, 255).astype(np.uint8)
            lut = ix
        lut = np.array(lut, dtype=np.uint8)

        # Vectorized implementation of above block.
        # This is overall slower.
        # h_nz = np.nonzero(h)[0]
        # if len(h_nz) <= 1:
        #     lut = np.arange(256).astype(np.uint8)
        # else:
        #     lo = h_nz[0]
        #     hi = h_nz[-1]
        #
        #     scale = 255.0 / (hi - lo)
        #     offset = -lo * scale
        #     ix = np.arange(256).astype(np.float64) * scale + offset
        #     ix = np.clip(ix, 0, 255).astype(np.uint8)
        #     lut = ix

        result[:, :, c_idx] = cv2.LUT(image_c, lut)
    if image.ndim == 2:
        return result[..., 0]
    return result


class PILSolarize(arithmetic.Solarize):
    """Augmenter with identical outputs to PIL's ``solarize()`` function.

    This augmenter inverts all pixel values above a threshold.

    This class is currently an alias for
    :class:`imgaug.augmenters.arithmetic.Solarize`, i.e. both classes are
    currently guarantueed to have the same outputs as PIL's function.

    dtype support::

        See :class:`imgaug.augmenters.arithmetic.Solarize`.

    """


class PILPosterize(color.Posterize):
    """Augmenter with identical outputs to PIL's ``posterize()`` function.

    This augmenter quantizes each array component to ``N`` bits.

    This class is currently an alias for
    :class:`imgaug.augmenters.color.Posterize`, which again is an alias
    for :class:`imgaug.augmenters.color.UniformColorQuantizationToNBits`,
    i.e. all three classes are right now guarantueed to have the same
    outputs as PIL's function.

    dtype support::

        See :class:`imgaug.augmenters.color.Posterize`.

    """


class PILEqualize(meta.Augmenter):
    """Equalize the image histogram.

    This augmenter has identical outputs to :func:`PIL.ImageOps.equalize`.

    dtype support::

        See :func:`imgaug.augmenters.pil.pil_equalize_`.

    Parameters
    ----------
    name : None or str, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    deterministic : bool, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    Examples
    --------
    >>> import imgaug.augmenters as iaa
    >>> aug = iaa.PILEqualize()

    Equalize the histograms of all input images.

    """
    def __init__(self, name=None, deterministic=False, random_state=None):
        super(PILEqualize, self).__init__(
            name=name, deterministic=deterministic, random_state=random_state)

    def _augment_batch(self, batch, random_state, parents, hooks):
        # pylint: disable=no-self-use
        if batch.images:
            for image in batch.images:
                image[...] = pil_equalize_(image)
        return batch

    def get_parameters(self):
        return []


class PILAutocontrast(contrast._ContrastFuncWrapper):
    """Adjust contrast by cutting off ``p%`` of lowest/highest histogram values.

    This augmenter has identical outputs to :func:`PIL.ImageOps.autocontrast`.

    See :func:`imgaug.augmenters.pil.pil_autocontrast` for more details.

    dtype support::

        See :func:`imgaug.augmenters.pil.pil_autocontrast`.

    Parameters
    ----------
    cutoff : int or tuple of int or list of int or imgaug.parameters.StochasticParameter, optional
        Percentage of values to cut off from the low and high end of each
        image's histogram, before stretching it to ``[0, 255]``.

            * If ``int``: The value will be used for all images.
            * If ``tuple`` ``(a, b)``: A value will be uniformly sampled from
              the discrete interval ``[a..b]`` per image.
            * If ``list``: A random value will be sampled from the list
              per image.
            * If ``StochasticParameter``: A value will be sampled from that
              parameter per image.

    per_channel :  bool or float, optional
        Whether to use the same value for all channels (``False``) or to
        sample a new value for each channel (``True``). If this value is a
        float ``p``, then for ``p`` percent of all images `per_channel` will
        be treated as ``True``, otherwise as ``False``.

    name : None or str, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    deterministic : bool, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    random_state : None or int or imgaug.random.RNG or numpy.random.Generator or numpy.random.BitGenerator or numpy.random.SeedSequence or numpy.random.RandomState, optional
        See :func:`imgaug.augmenters.meta.Augmenter.__init__`.

    Examples
    --------
    >>> import imgaug.augmenters as iaa
    >>> aug = iaa.PILAutocontrast()

    Modify the contrast of images by cutting off the ``0`` to ``20%`` lowest
    and highest values from the histogram, then stretching it to full length.

    >>> aug = iaa.PILAutocontrast((10, 20), per_channel=True)

    Modify the contrast of images by cutting off the ``10`` to ``20%`` lowest
    and highest values from the histogram, then stretching it to full length.
    The cutoff value is sampled per *channel* instead of per *image*.

    """
    def __init__(self, cutoff=(0, 20), per_channel=False,
                 name=None, deterministic=False, random_state=None):
        params1d = [
            iap.handle_discrete_param(
                cutoff, "cutoff", value_range=(0, 49), tuple_to_uniform=True,
                list_to_choice=True)
        ]
        func = pil_autocontrast

        super(PILAutocontrast, self).__init__(
            func, params1d, per_channel,
            dtypes_allowed=["uint8"],
            dtypes_disallowed=["uint16", "uint32", "uint64",
                               "int8", "int16", "int32", "int64",
                               "float16", "float32", "float64",
                               "float16", "float32", "float64", "float96",
                               "float128", "float256", "bool"],
            name=name,
            deterministic=deterministic,
            random_state=random_state
        )
