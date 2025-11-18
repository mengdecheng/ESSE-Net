# transforms.py
import albumentations as A
from albumentations.pytorch import ToTensorV2
import config  # 导入 config

def get_train_transform():
    return A.Compose([
        A.PadIfNeeded(min_height=config.IMG_HEIGHT, min_width=config.IMG_WIDTH, border_mode=0, value=0 ),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=30, p=0.5),
        A.ElasticTransform(alpha=8, sigma=50, p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.Resize(config.IMG_HEIGHT, config.IMG_WIDTH),
        ToTensorV2()
    ])

def get_val_transform():
    return A.Compose([
        A.PadIfNeeded(min_height=config.IMG_HEIGHT, min_width=config.IMG_WIDTH, border_mode=0, value=0),
        A.Resize(config.IMG_HEIGHT, config.IMG_WIDTH),
        ToTensorV2()
    ])