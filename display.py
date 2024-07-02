import matplotlib.pyplot as plt


def display(
    images, n=10, size=(20, 3), cmap="gray_r", as_type="float32", save_to=None
):
    """Function to plot images."""
    if images.max() > 1.0:
        images = images / 255.0
    elif images.min() < 0.0:
        images = (images + 1.0) / 2.0

    _, axes = plt.subplots(1, 10, figsize=size)
    axes = axes.ravel()

    for i in range(n):
        axes[i].imshow(images[i].astype(as_type), cmap=cmap)
        axes[i].axis("off")

    if save_to:
        plt.savefig(save_to)
        print(f"\nSaved to {save_to}")

    plt.show()
    plt.close()
