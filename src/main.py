from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms

# sudo apt install python3-tk
matplotlib.use("TkAgg")


DATA_CACHE = Path(__file__).parent.parent / "data/"
MNIST_MEAN_VAR = (0.1307, 0.3081)


def display_tensor_image(image: torch.Tensor) -> None:
    """Displays an image tensor (from zero mean unit variance dataset)."""
    # unnormalize the image
    image = image * MNIST_MEAN_VAR[1] + MNIST_MEAN_VAR[0]

    # display it
    plt.imshow(image.cpu().numpy().swapaxes(0, -1))
    plt.show()


def get_dataset(train: bool) -> Dataset:
    """Returns the MNIST dataset as a dataloader (so you can use iterators)."""
    # define transformation to normalize image to 0 mean and unit variance
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((MNIST_MEAN_VAR[0],), (MNIST_MEAN_VAR[1],)),
        ]
    )

    # return the dataloader
    return datasets.MNIST(
        root=DATA_CACHE,
        train=True,
        download=True,
        transform=transform,
    )


def main() -> None:
    for image, _ in get_dataset(train=True):
        print(image.shape)
        display_tensor_image(image)
        exit()


if __name__ == "__main__":
    main()
