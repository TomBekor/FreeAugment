# Augmentations magnidutes unnormalized bounds:
aug_bounds = {
    "Rotation": [-30, 30],
    "TranslateX": [-0.5, 0.5],
    "TranslateY": [-0.5, 0.5],
    "ShearX": [-0.6, 0.6],
    "ShearY": [-0.6, 0.6],
    "Contrast": [0.4, 2.0],
    "Sharpness": [0.0, 2.0],
    "Brightness": [-0.4, 0.4],
    "Solarize": [0.6, 1.0],
    "Color": [0.0, 1.0],
    "Posterize": [2.0, 8.0],
}

CUTOUT = 60

low_init = 0.125
high_init = 0.875
