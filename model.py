import torch
import torch.nn as nn
from unetPlus import UnetPlusPlus

class LiverSegmentationModel(nn.Module):
    def __init__(self, encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, classes=2):
        #encoder_weights="imagenet"预训练权重
        super().__init__()
        self.model = UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes
        )

    def forward(self, x):
        # 不再动态检查尺寸，直接假定为 256×256
        return self.model(x)
