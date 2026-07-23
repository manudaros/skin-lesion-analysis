import matplotlib.pyplot as plt
import numpy as np
import torch

from step12_data_augmentation import (
    LesionDataset,
    build_train_transform,
)


def denormalize(
    image_tensor: torch.Tensor,
) -> np.ndarray:
    """
    Reverse ImageNet normalization so an image can be displayed.

    Parameters
    ----------
    image_tensor:
        Normalized image tensor with shape [3, H, W].

    Returns
    -------
    np.ndarray
        Displayable RGB image with shape [H, W, 3]
        and values clipped to [0, 1].
    """
    image = (
        image_tensor
        .detach()
        .cpu()
        .permute(1, 2, 0)
        .numpy()
    )

    mean = np.array(
        [0.485, 0.456, 0.406],
        dtype=np.float32,
    )

    std = np.array(
        [0.229, 0.224, 0.225],
        dtype=np.float32,
    )

    image = (
        image * std
    ) + mean

    return np.clip(
        image,
        0.0,
        1.0,
    )


def show_augmentations(
    dataset: LesionDataset,
    image_index: int = 0,
    num_variations: int = 3,
) -> None:
    """
    Display several random augmentations of the same source image.

    Each row contains one augmented RGB image and its six
    spatially aligned segmentation masks.
    """
    if image_index < 0 or image_index >= len(dataset):
        raise IndexError(
            f"image_index must be between 0 and "
            f"{len(dataset) - 1}, received {image_index}."
        )

    if num_variations <= 0:
        raise ValueError(
            "num_variations must be greater than zero."
        )

    mask_keys = [
        "task1_segmentation",
        "pigment_network",
        "negative_network",
        "streaks",
        "milia_like_cysts",
        "globules",
    ]

    figure, axes = plt.subplots(
        nrows=num_variations,
        ncols=7,
        figsize=(
            20,
            3.2 * num_variations,
        ),
        squeeze=False,
    )

    source_image_id = dataset.data[
        image_index
    ]["image_id"]

    figure.suptitle(
        "Dynamic Augmentation Check\n"
        f"Image ID: {source_image_id}",
        fontsize=16,
    )

    for row_index in range(
        num_variations
    ):
        # Each access applies a newly sampled Torchvision v2 transform.
        sample = dataset[
            image_index
        ]

        image_array = denormalize(
            sample["image"]
        )

        image_axis = axes[
            row_index,
            0,
        ]

        image_axis.imshow(
            image_array
        )

        image_axis.set_title(
            f"Augmented Image "
            f"{row_index + 1}"
        )

        image_axis.axis(
            "off"
        )

        for (
            column_index,
            mask_key,
        ) in enumerate(
            mask_keys,
            start=1,
        ):
            mask_tensor = sample[
                mask_key
            ]

            mask_array = (
                mask_tensor
                .detach()
                .cpu()
                .squeeze(0)
                .numpy()
            )

            mask_axis = axes[
                row_index,
                column_index,
            ]

            mask_axis.imshow(
                mask_array,
                cmap="gray",
                vmin=0,
                vmax=1,
            )

            mask_axis.set_title(
                mask_key
                .replace("_", " ")
                .title(),
                fontsize=9,
            )

            mask_axis.axis(
                "off"
            )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    image_size = 384

    train_transform = (
        build_train_transform(
            image_size=image_size
        )
    )

    print(
        "Loading the training dataset..."
    )

    dataset = LesionDataset(
        split="train",
        transform=train_transform,
        include_task2=True,
    )

    print(
        f"Loaded {len(dataset)} "
        "training samples."
    )

    if len(dataset) > 0:
        print(
            "Displaying three random augmentations "
            "of the first training image..."
        )

        show_augmentations(
            dataset=dataset,
            image_index=0,
            num_variations=3,
        )