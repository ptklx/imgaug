# Added Module `imgaug.augmenters.pil` #479 #480

* Added module `imgaug.augmenters.pil`, which contains augmenters and
  functions corresponding to commonly used PIL functions. Their outputs
  are guaranteed to be identical to the PIL outputs.
* Added the following functions to the module:
  * `imgaug.augmenters.pil.pil_equalize`.
  * `imgaug.augmenters.pil.pil_equalize_`.
  * `imgaug.augmenters.pil.pil_autocontrast`.
  * `imgaug.augmenters.pil.pil_autocontrast_`.
  * `imgaug.augmenters.pil.pil_solarize`
  * `imgaug.augmenters.pil.pil_solarize_`
  * `imgaug.augmenters.pil.pil_posterize`.
  * `imgaug.augmenters.pil.pil_posterize_`.
  * `imgaug.augmenters.pil.pil_color`.
  * `imgaug.augmenters.pil.pil_contrast`.
  * `imgaug.augmenters.pil.pil_brightness`.
  * `imgaug.augmenters.pil.pil_sharpness`.
* Added the following augmenters to the module:
  * `imgaug.augmenters.pil.PILSolarize`.
  * `imgaug.augmenters.pil.PILPosterize`.
    (Currently alias for `imgaug.augmenters.color.Posterize`.)
  * `imgaug.augmenters.pil.PILEqualize`.
  * `imgaug.augmenters.pil.PILAutocontrast`.
