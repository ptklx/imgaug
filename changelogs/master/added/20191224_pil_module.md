# Added Module `imgaug.augmenters.pil` #479 #480 #537

* Added module `imgaug.augmenters.pil`, which contains augmenters and
  functions corresponding to commonly used PIL functions. Their outputs
  are guaranteed to be identical to the PIL outputs.
* Added the following functions to the module:
  * `imgaug.augmenters.pil.pil_equalize`.
  * `imgaug.augmenters.pil.pil_equalize_`.
  * `imgaug.augmenters.pil.pil_autocontrast`.
  * `imgaug.augmenters.pil.pil_autocontrast_`.
* Added the following augmenters to the module:
  * `imgaug.augmenters.pil.PILSolarize`.
    (Currently alias for `imgaug.augmenters.arithmetic.Solarize`.)
  * `imgaug.augmenters.pil.PILPosterize`.
    (Currently alias for `imgaug.augmenters.color.Posterize`.)
  * `imgaug.augmenters.pil.PILEqualize`.
  * `imgaug.augmenters.pil.PILAutocontrast`.
