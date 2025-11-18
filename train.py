import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
import matplotlib

matplotlib.use('Agg')  # 非交互式后端，避免GUI问题
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from tqdm import tqdm

from dataset import LiverSegmentationDataset
from transforms import get_train_transform, get_val_transform
from model import LiverSegmentationModel
from new_utils import predict_and_visualize
from config import *

# 设置日志文件路径SKA+AFF+MSSE
dataset_name = "resnet34(Kvasir-SEG)-MSSEindecoderCBAM"
log_file_path = f"{dataset_name}——Kvasir-SEGindecoderCBAM.txt"
#indecoder-SimAM  indecoder-CC-Module  indecoder- CBAM
# 配置 logging，记录日志到文件并输出到控制台
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file_path),
        logging.StreamHandler(sys.stdout)
    ]
)

logging.info("Training Log Start")
logging.info(f"Training with {EPOCHS} epochs and batch size {BATCH_SIZE}")

# 计算指标的辅助函数
def calculate_metrics(predictions, labels, num_classes=2):
    metrics = {
        "IoU": [0] * num_classes,
        "Precision": [0] * num_classes,
        "Recall": [0] * num_classes,
        "Dice(DSC)": [0] * num_classes,
        "Accuracy": [0] * num_classes,
    }
    total_pixels = labels.numel()

    for i in range(num_classes):
        TP = torch.logical_and(predictions == i, labels == i).sum().item()
        FP = torch.logical_and(predictions == i, labels != i).sum().item()
        FN = torch.logical_and(predictions != i, labels == i).sum().item()
        Total = (labels == i).sum().item()

        metrics["IoU"][i] = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0
        metrics["Precision"][i] = TP / (TP + FP) if (TP + FP) > 0 else 0
        metrics["Recall"][i] = TP / (TP + FN) if (TP + FN) > 0 else 0
        metrics["Dice(DSC)"][i] = 2 * TP / (2 * TP + FP + FN) if (2 * TP + FP + FN) > 0 else 0
        metrics["Accuracy"][i] = TP / Total if Total > 0 else 0

    return metrics

def plot_metrics(train_losses, val_losses, val_mIoUs, save_path="training_metrics.png"):
    plt.figure(figsize=(15, 5))

    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    # 绘制mIoU曲线
    plt.subplot(1, 2, 2)
    plt.plot(val_mIoUs, label='Validation mIoU', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('mIoU')
    plt.title('Validation Mean IoU')
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    logging.info(f"Training metrics plots saved to {save_path}")

# 创建数据集和 DataLoader
train_dataset = LiverSegmentationDataset(
    image_dir=TRAIN_IMAGES_PATH,
    mask_dir=TRAIN_MASKS_PATH,
    transform=get_train_transform()
)

val_dataset = LiverSegmentationDataset(
    image_dir=VAL_IMAGES_PATH,
    mask_dir=VAL_MASKS_PATH,
    transform=get_val_transform()
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# 初始化模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LiverSegmentationModel().to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

# 初始化指标存储
train_losses = []
val_losses = []
val_mIoUs = []
best_mIoU = 0.80
best_model_path = ""
best_epoch = 0

# 图像归一化的均值和标准差（Kvasir-SEG）
MEAN = [0.5571526211500167, 0.3217285327464342, 0.23587989364936948]
STD = [0.3180630469851302, 0.2214476214183649, 0.18723953023747872]

# 训练循环
for epoch in range(EPOCHS):
    logging.info(f"-------- Epoch {epoch + 1} --------")
    running_loss = 0.0

    # 训练阶段
    model.train()
    for images, masks in tqdm(train_loader, desc="Training", unit="batch"):
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_train_loss = running_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    logging.info(f"Train Loss: {avg_train_loss:.4f}")

    # 验证阶段
    model.eval()
    val_loss = 0.0
    val_metrics = {
        "IoU": [0] * Class,
        "Precision": [0] * Class,
        "Recall": [0] * Class,
        "Dice(DSC)": [0] * Class,
        "Accuracy": [0] * Class,
    }
    num_batches = len(val_loader)

    with torch.no_grad():
        for images, masks in tqdm(val_loader, desc="Validation", unit="batch"):
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            val_loss += loss.item()

            probabilities = torch.softmax(outputs, dim=1)
            predicted_labels = torch.argmax(probabilities, dim=1)
            batch_metrics = calculate_metrics(predicted_labels, masks, num_classes=Class)

            for key in val_metrics:
                for i in range(Class):
                    val_metrics[key][i] += batch_metrics[key][i]

    # 计算平均指标
    final_metrics = {
        key: [value / num_batches for value in val_metrics[key]]
        for key in val_metrics
    }

    avg_val_loss = val_loss / len(val_loader)
    val_losses.append(avg_val_loss)

    avg_mIoU = sum(final_metrics["IoU"]) / len(final_metrics["IoU"])
    val_mIoUs.append(avg_mIoU)

    logging.info(f"Validation Loss: {avg_val_loss:.4f}")
    logging.info(f"Mean IoU (mIoU): {avg_mIoU:.4f}")
    for i in range(Class):
        logging.info(f"Class {i} - IoU: {final_metrics['IoU'][i]:.4f}, "
                     f"Precision: {final_metrics['Precision'][i]:.4f}, "
                     f"Recall: {final_metrics['Recall'][i]:.4f}, "
                     f"Dice(DSC): {final_metrics['Dice(DSC)'][i]:.4f}, "
                     f"Accuracy: {final_metrics['Accuracy'][i]:.4f}")

    # 保存最佳模型（不覆盖之前的模型）
    if avg_mIoU > best_mIoU:
        best_mIoU = avg_mIoU
        best_epoch = epoch + 1
        # 生成唯一的模型文件名
        model_path = f"{dataset_name}_epoch_{best_epoch}_mIoU_{best_mIoU:.4f}.pth"
        torch.save(model.state_dict(), model_path)
        logging.info(f"New best model saved: {model_path} at epoch {best_epoch}")
        # 更新best_model_path以便后续使用（如可视化）
        best_model_path = model_path

# 绘制训练曲线
plot_metrics(
    train_losses,
    val_losses,
    val_mIoUs,
    save_path=f"{dataset_name}_training_metrics.png"
)

# 训练完成后可视化
if best_model_path:
    logging.info(f"Loading best model from {best_model_path}")
    model.load_state_dict(torch.load(best_model_path))
    model.to(device)
    model.eval()

    visualize_dataset = LiverSegmentationDataset(
        image_dir=VAL_IMAGES_PATH,
        mask_dir=VAL_MASKS_PATH,
        transform=get_val_transform()
    )

    selected_indices = list(range(10))
    selected_dataset = torch.utils.data.Subset(visualize_dataset, selected_indices)
    selected_loader = DataLoader(selected_dataset, batch_size=10, shuffle=False, num_workers=0)

    predict_and_visualize(
        model=model,
        dataloader=selected_loader,
        device=device,
        epoch=best_epoch,
        save_dir="final_visualizations",
        num_batches=1,
        images_per_batch=10,
        num_classes=2,
        mean=MEAN,
        std=STD,
        target_size=(512, 512)
    )

    logging.info("Final visualizations saved in 'final_visualizations' directory.")

logging.info(f"Training complete. Best mIoU: {best_mIoU:.4f} at epoch {best_epoch}, Last best model saved at: {best_model_path}")