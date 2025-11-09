import json, cv2, numpy as np, torch, albumentations as A

def get_transforms(train=True, img_size=1024):
    if train:
        tfms = [
            A.LongestMaxSize(max_size=img_size),
            A.PadIfNeeded(img_size, img_size, border_mode=cv2.BORDER_CONSTANT, value=(114,114,114)),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.02, scale_limit=0.10, rotate_limit=10,
                               border_mode=cv2.BORDER_CONSTANT, value=(114,114,114), p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.CLAHE(p=0.2),
        ]
    else:
        # tfms = [
        #     A.LongestMaxSize(max_size=img_size),
        #     A.PadIfNeeded(img_size, img_size, border_mode=cv2.BORDER_CONSTANT, value=(114,114,114)),
        # ]
        return None
    return A.Compose(
        tfms,
        bbox_params=A.BboxParams(
            format="pascal_voc",
            label_fields=["class_labels"],
            min_visibility=0.0,
        )
    )
