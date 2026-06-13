"""Random weather degradation for training data augmentation.

Applies randomly-selected corruptions to a fraction of training images,
controlled by a CLI flag in yolo26_train.py.
"""

import random
import cv2
import albumentations as A

# Fraction of training images that receive each degradation type.
# Total degraded: 0.30 (30%)
_DEGRADE_WEIGHTS = {
    "snow": 0.08,
    "frost": 0.08,
    "fog": 0.08,
    "rain": 0.03,
    "brightness": 0.03,
}
_DEGRADE_TOTAL = sum(_DEGRADE_WEIGHTS.values())  # 0.30

_SEVERITY = 3


def _build_transform(degrade_type: str) -> A.Compose:
    """Create an albumentations Compose for a single degradation type at fixed severity."""
    alpha = _SEVERITY / 5.0
    if degrade_type == "snow":
        return A.Compose([
            A.RandomSnow(p=1.0, snow_point_range=(0.1, 0.3 + 0.2 * alpha),
                         brightness_coeff=0.5 + 0.5 * alpha),
        ])
    elif degrade_type == "frost":
        return A.Compose([
            A.RandomBrightnessContrast(
                brightness_limit=(-0.3 * alpha, 0),
                contrast_limit=(-0.3 * alpha, 0), p=1.0),
        ])
    elif degrade_type == "fog":
        return A.Compose([
            A.RandomFog(p=1.0, fog_coef=alpha, alpha_coef=alpha),
        ])
    elif degrade_type == "rain":
        return A.Compose([
            A.RandomRain(p=1.0, drop_width=1, drop_length=20,
                         drop_width_range=(1, 2), blur_value=1),
        ])
    elif degrade_type == "brightness":
        return A.Compose([
            A.RandomBrightnessContrast(
                brightness_limit=(-0.3 * alpha, 0), contrast_limit=0, p=1.0),
        ])
    else:
        return A.Compose([])


def apply_random_degradation(img):
    """Apply a random degradation to a BGR image with 30% probability.

    Within the 30%, the degradation type is sampled according to:
        snow 8%, frost 8%, fog 8%, rain 3%, brightness 3%.

    Args:
        img: numpy array (H, W, 3) in BGR format.

    Returns:
        numpy array (H, W, 3) in BGR format (new array if degraded, else original).
    """
    if random.random() >= _DEGRADE_TOTAL:
        return img

    types = list(_DEGRADE_WEIGHTS.keys())
    weights = list(_DEGRADE_WEIGHTS.values())
    degrade_type = random.choices(types, weights=weights, k=1)[0]

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    corrupted = _build_transform(degrade_type)(image=img_rgb)["image"]
    return cv2.cvtColor(corrupted, cv2.COLOR_RGB2BGR)
