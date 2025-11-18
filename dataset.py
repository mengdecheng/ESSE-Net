import os
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torch
import warnings
warnings.filterwarnings("ignore", message="To copy construct from a tensor")

class LiverSegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        """
        初始化数据集
        :param image_dir: 图像文件夹路径
        :param mask_dir: 掩码文件夹路径
        :param transform: 图像和掩码的预处理变换
        """

        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_names = os.listdir(image_dir)
        self.image_paths = [os.path.join(self.image_dir, name) for name in self.image_names]
        self.transform = transform

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_names[idx])
        mask_path = os.path.join(self.mask_dir, self.image_names[idx])

        # 加载图像和掩码
        image = Image.open(img_path).convert("RGB")  # 确保图像为 RGB 格式
        mask = Image.open(mask_path).convert("L")  # 掩码为灰度图

        # 数据增强,罪魁祸首！！！！！！！！！！！！！！！！！！！！！！！！W与C换位置了
        if self.transform:
            augmented = self.transform(image=np.array(image), mask=np.array(mask))
            image = augmented["image"]  # [H, W, C]
            mask = augmented["mask"]  # [H, W]
        # 确保 mask 的值在有效范围内!!!!!!!!!!!!!!!!!!!!换数据集的时候记得改max
        # mask = torch.clamp(mask, min=0, max=2)
        # 二值化掩码处理：背景为 0，其他为 1
        mask = np.where(mask == 0, 0, 1)  # 将背景设为 0，其他类设为 1
        # 转换为 PyTorch 张量并调整维度
        image = torch.tensor(image, dtype=torch.float32).contiguous().clone().detach()
        mask = torch.tensor(mask, dtype=torch.long).contiguous().clone().detach()

        # print(f"Image shape: {image.shape}")  # 调试信息，应为 [C, H, W]
        # print(f"Mask shape: {mask.shape}")  # 调试信息，应为 [H, W]

        return image, mask
