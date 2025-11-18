# new_utils.py

import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.colors as colors
import cv2

def denormalize(image, mean, std):
    """
    Denormalizes an image tensor.

    :param image: Tensor of shape (C, H, W)
    :param mean: List of means for each channel
    :param std: List of standard deviations for each channel
    :return: Denormalized image tensor
    """
    for t, m, s in zip(image, mean, std):
        t.mul_(s).add_(m)
    return image

def resize_image(image, target_size=(512, 512), interpolation=cv2.INTER_AREA):
    """
    Resizes an image to the target size.

    :param image: NumPy array of the image
    :param target_size: Tuple specifying the target size (width, height)
    :param interpolation: Interpolation method
    :return: Resized image as a NumPy array
    """
    if image.size == 0:
        raise ValueError("Empty image cannot be resized.")
    return cv2.resize(image, target_size, interpolation=interpolation)

def overlay_mask(image, mask, alpha=0.5, cmap='jet'):
    """
    Overlays a mask on an image with transparency.

    :param image: Original image as a NumPy array (H, W, C)
    :param mask: Mask image as a NumPy array (H, W)
    :param alpha: Transparency factor for the mask overlay
    :param cmap: Colormap for the mask
    :return: Image with mask overlay
    """
    if mask.ndim == 2:
        # Normalize mask to range [0, num_classes-1]
        mask_colored = plt.get_cmap(cmap)(mask)[:, :, :3]  # Convert to RGB
    else:
        mask_colored = mask  # Assume mask is already colored

    overlayed_image = (1 - alpha) * image + alpha * mask_colored
    overlayed_image = np.clip(overlayed_image, 0, 1)
    return overlayed_image

def highlight_edges(image, mask, color=(0, 255, 0)):
    """
    Highlights the edges of the mask on the image.

    :param image: Original image as a NumPy array (H, W, C)
    :param mask: Mask image as a NumPy array (H, W)
    :param color: Color for the edges (R, G, B)
    :return: Image with highlighted edges
    """
    # Convert mask to binary
    mask_binary = (mask > 0).astype(np.uint8) * 255
    edges = cv2.Canny(mask_binary, 100, 200)

    # Create an edge image with the specified color
    edge_image = np.zeros_like(image)
    edge_image[edges != 0] = np.array(color) / 255.0  # Normalize color

    # Overlay edges on the original image
    highlighted_image = cv2.addWeighted(image, 1, edge_image, 1, 0)
    return highlighted_image


def predict_and_visualize(model, dataloader, device, epoch, save_dir="visualizations",
                         num_batches=1, images_per_batch=10, num_classes=2,
                         mean=None, std=None, target_size=(512, 512), boundary_color=255):
    """
    使用模型对数据进行预测，将真实蒙版与预测蒙版拼接，并在中间添加边界线，解决显示过暗问题。

    :param model: 训练好的模型
    :param dataloader: 包含图像和掩码的 DataLoader
    :param device: 模型运行的设备 (CPU 或 GPU)
    :param epoch: 当前的 epoch 数，用于保存文件命名
    :param save_dir: 保存预测结果的目录
    :param num_batches: 需要可视化的批次数量
    :param images_per_batch: 每个批次中需要可视化的图像数量
    :param num_classes: 分割类别数
    :param mean: 图像归一化的均值（用于反归一化）
    :param std: 图像归一化的标准差（用于反归一化）
    :param target_size: 目标图像大小 (width, height) 用于调整图像大小
    :param boundary_color: 边界线颜色（默认为白色 255）
    """
    model.eval()  # 切换到评估模式

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with torch.no_grad():  # 在预测过程中不计算梯度
        for batch_idx, (images, masks) in enumerate(dataloader):
            if batch_idx >= num_batches:
                break

            images, masks = images.to(device), masks.to(device)

            # 获取模型的预测输出
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_labels = torch.argmax(probabilities, dim=1)

            # 可视化
            for i in range(min(len(images), images_per_batch)):
                # 将预测掩码转换为 numpy 数组
                predicted_mask = predicted_labels[i].cpu().numpy()
                real_mask = masks[i].cpu().numpy()

                # 确保 mask 是 2D
                if real_mask.ndim == 3 and real_mask.shape[0] == 1:
                    real_mask = real_mask.squeeze(0)
                if predicted_mask.ndim == 3 and predicted_mask.shape[0] == 1:
                    predicted_mask = predicted_mask.squeeze(0)

                # 调整蒙版大小
                try:
                    real_mask_resized = resize_image(real_mask, target_size=target_size, interpolation=cv2.INTER_NEAREST)
                    predicted_mask_resized = resize_image(predicted_mask, target_size=target_size, interpolation=cv2.INTER_NEAREST)
                except ValueError as e:
                    print(f"Error resizing mask in epoch {epoch+1}, batch {batch_idx+1}, image {i+1}: {e}")
                    continue

                # 归一化蒙版到 [0, 255] 范围
                if real_mask_resized.max() == 0:
                    real_mask_normalized = np.zeros_like(real_mask_resized, dtype=np.uint8)
                else:
                    real_mask_normalized = (real_mask_resized / real_mask_resized.max() * 255).astype(np.uint8)
                if predicted_mask_resized.max() == 0:
                    predicted_mask_normalized = np.zeros_like(predicted_mask_resized, dtype=np.uint8)
                else:
                    predicted_mask_normalized = (predicted_mask_resized / predicted_mask_resized.max() * 255).astype(np.uint8)

                # 添加边界线
                height, width = real_mask_normalized.shape
                boundary_line = np.ones((height, 5)) * boundary_color  # 5像素宽度的白色边界线

                # 将真实蒙版、边界线和预测蒙版水平拼接
                combined_mask = np.concatenate([real_mask_normalized, boundary_line, predicted_mask_normalized], axis=1)

                # 保存拼接后的图像
                plt.figure(figsize=(12, 6))
                plt.imshow(combined_mask, cmap="gray")  # 显示为灰度图
                plt.title(f"True Mask (Left) vs Predicted Mask (Right) - Batch {batch_idx+1} Image {i+1}")
                plt.axis("off")
                combined_mask_filename = f"epoch_{epoch+1}_batch_{batch_idx+1}_image_{i+1}_combined_mask.png"
                plt.savefig(os.path.join(save_dir, combined_mask_filename), dpi=300)
                plt.close()

    print(f"Visualizations saved in directory: {save_dir}")

def resize_image(image, target_size=(512, 512), interpolation=cv2.INTER_NEAREST):
    """
    调整图像大小。
    :param image: 输入图像
    :param target_size: 目标大小 (width, height)
    :param interpolation: 插值方法
    :return: 调整大小后的图像
    """
    return cv2.resize(image, target_size, interpolation=interpolation)