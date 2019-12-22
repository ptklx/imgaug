# Added Wrappers for `imagecorruptions` Package #530

Added wrappers around the functions from package
[bethgelab/imagecorruptions](https://github.com/bethgelab/imagecorruptions).
The functions in that package were used in some recent papers and are added
here for convenience.
The wrappers produce arrays containing values identical to the output
arrays from the corresponding `imagecorruptions` functions when called
via the `imagecorruptions.corrupt()` (verified via unittests).
The interfaces of the wrapper functions are identical to the
`imagecorruptions` functions, with the only difference of also supporting
`seed` parameters.

Note that to ensure identical outputs the implemented functions always wrap
functions from `imagecorruptions`, even when faster and more robust
alternatives exist in `imgaug`.

* Added module `imgaug.augmenters.imgcorrupt`.
* Added the following functions to module `imgaug.augmenters.imgcorrupt`:
    * `apply_imgcorrupt_gaussian_noise()`
    * `apply_imgcorrupt_shot_noise()`
    * `apply_imgcorrupt_impulse_noise()`
    * `apply_imgcorrupt_speckle_noise()`
    * `apply_imgcorrupt_gaussian_blur()`
    * `apply_imgcorrupt_glass_blur()`
    * `apply_imgcorrupt_defocus_blur()`
    * `apply_imgcorrupt_motion_blur()`
    * `apply_imgcorrupt_zoom_blur()`
    * `apply_imgcorrupt_fog()`
    * `apply_imgcorrupt_snow()`
    * `apply_imgcorrupt_spatter()`
    * `apply_imgcorrupt_contrast()`
    * `apply_imgcorrupt_brightness()`
    * `apply_imgcorrupt_saturate()`
    * `apply_imgcorrupt_jpeg_compression()`
    * `apply_imgcorrupt_pixelate()`
    * `apply_imgcorrupt_elastic_transform()`
* Added function `imgaug.augmenters.imgcorrupt.get_imgcorrupt_subset(subset)`.
  Similar to `imgcorruptions.get_corruption_names(subset)`, but returns a
  tuple
  `(list of corruption method names, list of corruption method functions)`,
  instead of only the names.
* Added the following augmenters to module `imgaug.augmenters.imgcorrupt`:
    * `ImgcorruptGaussianNoise`
    * `ImgcorruptShotNoise`
    * `ImgcorruptImpulseNoise`
    * `ImgcorruptSpeckleNoise`
    * `ImgcorruptGaussianBlur`
    * `ImgcorruptGlassBlur`
    * `ImgcorruptDefocusBlur`
    * `ImgcorruptMotionBlur`
    * `ImgcorruptZoomBlur`
    * `ImgcorruptFog`
    * `ImgcorruptFrost`
    * `ImgcorruptSnow`
    * `ImgcorruptSpatter`
    * `ImgcorruptContrast`
    * `ImgcorruptBrightness`
    * `ImgcorruptSaturate`
    * `ImgcorruptJpegCompression`
    * `ImgcorruptPixelate`
    * `ImgcorruptElasticTransform`
* Added context `imgaug.random.temporary_numpy_seed()`.
