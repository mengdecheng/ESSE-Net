import matplotlib.pyplot as plt

def visualize_batch(images, masks):
    """
    可视化一个批次的图像和掩码
    """
    plt.figure(figsize=(12, 6))
    for i in range(min(len(images), 4)):
        plt.subplot(2, 4, i + 1)
        plt.imshow(images[i].permute(1, 2, 0).numpy())
        plt.title("Image")
        plt.axis("off")

        plt.subplot(2, 4, i + 5)
        plt.imshow(masks[i].numpy(), cmap="gray")
        plt.title("Mask")
        plt.axis("off")

    plt.tight_layout()
    plt.show()
