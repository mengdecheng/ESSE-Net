import os

# 数据路径配置
# BASE_PATH = "/home/mdc/桌面/mdc/不服气20241027/newnewLITS"
BASE_PATH = "../Kvasir-SEG"
# BASE_PATH = "../LIDC_IDRI"
# BASE_PATH = "../CVC-ClinicDB"
TRAIN_IMAGES_PATH = os.path.join(BASE_PATH, "train_images")
TRAIN_MASKS_PATH = os.path.join(BASE_PATH, "train_masks")
VAL_IMAGES_PATH = os.path.join(BASE_PATH, "val_images")
VAL_MASKS_PATH = os.path.join(BASE_PATH, "val_masks")

# 图像参数/Kvasir-SEG CVC-ColonDB 256*256/CVC-ClinicDB
IMG_HEIGHT = 256
IMG_WIDTH = 256
# 训练参数 Kvasir-SEG 36
BATCH_SIZE = 2
EPOCHS = 200
LEARNING_RATE = 0.0001
Class = 2