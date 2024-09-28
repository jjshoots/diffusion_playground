from utils import display_tensor_image, get_dataset


def main() -> None:
    for image, _ in get_dataset(train=True):
        print(image.shape)
        display_tensor_image(image)
        exit()


if __name__ == "__main__":
    main()
