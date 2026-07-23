import os
import csv

import pandas as pd
import torch
from torch import nn
from torch.utils.data import (
    DataLoader,
    Dataset,
)
from torchvision import tv_tensors
from torchvision.io import (
    ImageReadMode,
    read_image,
)
from torchvision.transforms import (
    InterpolationMode,
)
from torchvision.transforms import v2
from torchvision.transforms.v2 import (
    functional as F,
)


# ============================================================
# Deterministic resize and padding
# ============================================================

class ResizeLongestSideAndPad(nn.Module):
    """
    Resize an image and its masks while preserving aspect ratio,
    then pad them to a fixed square size.
    """

    def __init__(
        self,
        output_size: int = 384,
    ) -> None:
        super().__init__()

        if output_size <= 0:
            raise ValueError(
                "output_size must be greater than zero."
            )

        self.output_size = output_size

    def forward(
        self,
        image: tv_tensors.Image,
        masks: tv_tensors.Mask,
    ) -> tuple[
        tv_tensors.Image,
        tv_tensors.Mask,
    ]:
        """
        Apply identical resizing and padding to the image and masks.
        """
        image_height = int(
            image.shape[-2]
        )
        image_width = int(
            image.shape[-1]
        )

        scale = min(
            self.output_size / image_height,
            self.output_size / image_width,
        )

        new_height = max(
            1,
            int(round(
                image_height * scale
            )),
        )

        new_width = max(
            1,
            int(round(
                image_width * scale
            )),
        )

        # Use bilinear interpolation for the RGB image.
        image = F.resize(
            image,
            size=[
                new_height,
                new_width,
            ],
            interpolation=(
                InterpolationMode.BILINEAR
            ),
            antialias=True,
        )

        # Use nearest-neighbor interpolation for segmentation masks.
        masks = F.resize(
            masks,
            size=[
                new_height,
                new_width,
            ],
            interpolation=(
                InterpolationMode.NEAREST
            ),
        )

        horizontal_padding = (
            self.output_size
            - new_width
        )

        vertical_padding = (
            self.output_size
            - new_height
        )

        padding_left = (
            horizontal_padding
            // 2
        )

        padding_right = (
            horizontal_padding
            - padding_left
        )

        padding_top = (
            vertical_padding
            // 2
        )

        padding_bottom = (
            vertical_padding
            - padding_top
        )

        padding = [
            padding_left,
            padding_top,
            padding_right,
            padding_bottom,
        ]

        # Pad the image with an approximate ImageNet mean colour.
        image = F.pad(
            image,
            padding=padding,
            fill=[
                124,
                116,
                104,
            ],
        )

        # Padding areas in masks must remain background.
        masks = F.pad(
            masks,
            padding=padding,
            fill=0,
        )

        return image, masks


# ============================================================
# Transform construction
# ============================================================

def build_train_transform(
    image_size: int = 384,
) -> v2.Compose:
    """
    Build the training preprocessing and augmentation pipeline.
    """
    return v2.Compose(
        [
            ResizeLongestSideAndPad(
                output_size=image_size,
            ),

            v2.RandomHorizontalFlip(
                p=0.5,
            ),

            v2.RandomVerticalFlip(
                p=0.5,
            ),

            # Apply affine augmentation to only half of samples.
            v2.RandomApply(
                [
                    v2.RandomAffine(
                        degrees=(-15, 15),
                        translate=(
                            0.05,
                            0.05,
                        ),
                        scale=(
                            0.95,
                            1.05,
                        ),
                        interpolation=(
                            InterpolationMode.BILINEAR
                        ),
                        fill={
                            tv_tensors.Image: [
                                124,
                                116,
                                104,
                            ],
                            tv_tensors.Mask: 0,
                        },
                    )
                ],
                p=0.5,
            ),

            # Convert the image from uint8 [0, 255]
            # to float32 [0, 1].
            # Masks remain float32 binary tensors.
            v2.ToDtype(
                dtype={
                    tv_tensors.Image: (
                        torch.float32
                    ),
                    tv_tensors.Mask: (
                        torch.float32
                    ),
                    "others": None,
                },
                scale=True,
            ),

            # Apply ImageNet normalization to the image only.
            v2.Normalize(
                mean=[
                    0.485,
                    0.456,
                    0.406,
                ],
                std=[
                    0.229,
                    0.224,
                    0.225,
                ],
            ),

            # Convert TVTensors back to standard PyTorch tensors.
            v2.ToPureTensor(),
        ]
    )


def build_val_transform(
    image_size: int = 384,
) -> v2.Compose:
    """
    Build deterministic validation preprocessing.

    Validation data must not receive random augmentation.
    """
    return v2.Compose(
        [
            ResizeLongestSideAndPad(
                output_size=image_size,
            ),

            v2.ToDtype(
                dtype={
                    tv_tensors.Image: (
                        torch.float32
                    ),
                    tv_tensors.Mask: (
                        torch.float32
                    ),
                    "others": None,
                },
                scale=True,
            ),

            v2.Normalize(
                mean=[
                    0.485,
                    0.456,
                    0.406,
                ],
                std=[
                    0.229,
                    0.224,
                    0.225,
                ],
            ),

            v2.ToPureTensor(),
        ]
    )


# ============================================================
# Dataset
# ============================================================

class LesionDataset(Dataset):
    def __init__(
        self,
        fold: int = 0,
        role: str = "train",
        transform=None,
        include_task2: bool = True,
        fold_column: str = "fold",
    ) -> None:
        """
        Load Task 1 lesion masks and optionally Task 2 masks,
        selecting samples by cross-validation fold.

        Parameters
        ----------
        fold:
            Which fold number to treat as the held-out fold.

        role:
            Either "train" or "val".
            "val"   -> images whose fold == `fold`.
            "train" -> all images whose fold != `fold`.

        transform:
            A Torchvision v2 transform applied jointly to the image
            and all loaded masks.

        include_task2:
            When False, only the Task 1 lesion mask is loaded.
            This avoids unnecessary disk access during Task 1 training.

        fold_column:
            Name of the fold column in task1_task2_folds.csv.
        """
        if role not in {
            "train",
            "val",
        }:
            raise ValueError(
                "role must be either 'train' or 'val'."
            )

        self.script_dir = os.path.dirname(
            os.path.abspath(__file__)
        )

        self.project_root = os.path.abspath(
            os.path.join(
                self.script_dir,
                "..",
            )
        )

        self.data_root = os.path.join(
            self.project_root,
            "data",
        )

        self.csv_path = os.path.join(
            self.project_root,
            "index.csv",
        )

        self.fold_csv_path = os.path.join(
            self.project_root,
            "splits",
            "task1_task2_folds.csv",
        )

        self.fold = fold
        self.role = role
        self.transform = transform
        self.include_task2 = include_task2

        if not os.path.exists(
            self.csv_path
        ):
            raise FileNotFoundError(
                "Dataset index was not found: "
                f"{self.csv_path}"
            )

        if not os.path.exists(
            self.fold_csv_path
        ):
            raise FileNotFoundError(
                "Fold assignment file was not found: "
                f"{self.fold_csv_path}\n"
                "Run the fold-creation step first."
            )

        # ----- select image IDs for this fold and role -----

        folds = pd.read_csv(
            self.fold_csv_path,
            dtype={"image_id": str},
        )

        if fold_column not in folds.columns:
            raise ValueError(
                f"Column '{fold_column}' not found in "
                f"{self.fold_csv_path}. "
                f"Available columns: {list(folds.columns)}"
            )

        available_folds = sorted(
            folds[fold_column].unique().tolist()
        )

        if fold not in available_folds:
            raise ValueError(
                f"Fold {fold} not present in the fold file. "
                f"Available folds: {available_folds}"
            )

        if role == "val":
            selected = folds[
                folds[fold_column] == fold
            ]
        else:
            selected = folds[
                folds[fold_column] != fold
            ]

        self.valid_ids = {
            str(image_id)
            for image_id in selected["image_id"]
        }

        if not self.valid_ids:
            raise ValueError(
                f"No image IDs selected for fold={fold}, "
                f"role='{role}'."
            )

        # ----- join against index.csv for the file paths -----

        self.data = []

        with open(
            self.csv_path,
            mode="r",
            newline="",
            encoding="utf-8",
        ) as csv_file:
            reader = csv.DictReader(
                csv_file
            )

            for row in reader:
                image_id = row.get(
                    "image_id",
                    "",
                )

                if image_id in self.valid_ids:
                    self.data.append(
                        row
                    )

        found_ids = {
            row["image_id"]
            for row in self.data
        }

        missing_ids = (
            self.valid_ids
            - found_ids
        )

        if missing_ids:
            raise ValueError(
                "Some fold IDs were not found in index.csv. "
                f"Examples: {sorted(missing_ids)[:10]}"
            )

        if not self.data:
            raise ValueError(
                f"No samples were loaded for fold={fold}, "
                f"role='{role}'."
            )

    def __len__(self) -> int:
        """Return the number of samples."""
        return len(
            self.data
        )

    def __getitem__(
        self,
        idx: int,
    ) -> dict[str, object]:
        """Load one image and its corresponding masks."""
        row = self.data[idx]

        image_path = os.path.join(
            self.data_root,
            row["task1_image_path"],
        )

        if not os.path.exists(
            image_path
        ):
            raise FileNotFoundError(
                f"Image file was not found: "
                f"{image_path}"
            )

        # Load the RGB image as a uint8 tensor with shape [3, H, W].
        image = read_image(
            image_path,
            ImageReadMode.RGB,
        )

        image = tv_tensors.Image(
            image
        )

        mask_keys = [
            "task1_mask_path",
        ]

        if self.include_task2:
            mask_keys.extend(
                [
                    "pigment_network_mask",
                    "negative_network_mask",
                    "streaks_mask",
                    "milia_like_cysts_mask",
                    "globules_mask",
                ]
            )

        mask_list = []

        for key in mask_keys:
            mask_path = os.path.join(
                self.data_root,
                row[key],
            )

            if not os.path.exists(
                mask_path
            ):
                raise FileNotFoundError(
                    f"Mask file was not found: "
                    f"{mask_path}"
                )

            mask = read_image(
                mask_path,
                ImageReadMode.GRAY,
            )

            mask_list.append(
                mask
            )

        # Stack masks into [number_of_masks, H, W].
        stacked_masks = torch.cat(
            mask_list,
            dim=0,
        )

        # Support both 0/1 masks and 0/255 masks.
        stacked_masks = (
            stacked_masks > 0
        ).to(
            torch.float32
        )

        stacked_masks = tv_tensors.Mask(
            stacked_masks
        )

        if self.transform is not None:
            (
                image,
                stacked_masks,
            ) = self.transform(
                image,
                stacked_masks,
            )

        # Ensure standard tensors and exact binary mask values.
        image = torch.as_tensor(
            image
        ).contiguous()

        stacked_masks = (
            torch.as_tensor(
                stacked_masks
            )
            > 0.5
        ).to(
            torch.float32
        ).contiguous()

        sample = {
            "image_id": row[
                "image_id"
            ],
            "image": image,

            # Preserve the channel dimension:
            # [1, H, W] rather than [H, W].
            "task1_segmentation": (
                stacked_masks[0:1]
            ),
        }

        if self.include_task2:
            sample.update(
                {
                    # Combined Task 2 target:
                    # [5, H, W]
                    "task2_attributes": (
                        stacked_masks[1:6]
                    ),

                    # Individual attributes:
                    # each has shape [1, H, W]
                    "pigment_network": (
                        stacked_masks[1:2]
                    ),
                    "negative_network": (
                        stacked_masks[2:3]
                    ),
                    "streaks": (
                        stacked_masks[3:4]
                    ),
                    "milia_like_cysts": (
                        stacked_masks[4:5]
                    ),
                    "globules": (
                        stacked_masks[5:6]
                    ),
                }
            )

        return sample


# ============================================================
# Dataset and DataLoader test
# ============================================================

if __name__ == "__main__":
    image_size = 384
    dev_fold = 0

    train_transform = (
        build_train_transform(
            image_size=image_size,
        )
    )

    val_transform = (
        build_val_transform(
            image_size=image_size,
        )
    )

    print(
        "Initializing datasets "
        f"(development fold = {dev_fold})..."
    )

    # Only Task 1 masks are needed for the current Task 1 baseline.
    train_dataset = LesionDataset(
        fold=dev_fold,
        role="train",
        transform=train_transform,
        include_task2=False,
    )

    val_dataset = LesionDataset(
        fold=dev_fold,
        role="val",
        transform=val_transform,
        include_task2=False,
    )

    print(
        f"Loaded {len(train_dataset)} "
        "training samples."
    )

    print(
        f"Loaded {len(val_dataset)} "
        "validation samples."
    )

    # Guard against any accidental overlap between train and val.
    train_ids = {
        row["image_id"]
        for row in train_dataset.data
    }
    val_ids = {
        row["image_id"]
        for row in val_dataset.data
    }
    overlap = train_ids & val_ids

    print(
        f"Train/val overlap (must be 0): "
        f"{len(overlap)}"
    )

    if overlap:
        raise RuntimeError(
            "Train and validation sets overlap: "
            f"{sorted(overlap)[:10]}"
        )

    sample = train_dataset[0]

    print(
        "\nSingle-sample test:"
    )

    print(
        f"Image ID: "
        f"{sample['image_id']}"
    )

    print(
        f"Image tensor shape: "
        f"{tuple(sample['image'].shape)}"
    )

    print(
        f"Image tensor dtype: "
        f"{sample['image'].dtype}"
    )

    print(
        f"Task 1 mask shape: "
        f"{tuple(sample['task1_segmentation'].shape)}"
    )

    print(
        f"Task 1 mask dtype: "
        f"{sample['task1_segmentation'].dtype}"
    )

    print(
        "Task 1 mask unique values: "
        f"{torch.unique(sample['task1_segmentation']).tolist()}"
    )

    # A real DataLoader test is required to confirm that samples
    # can be stacked into a batch.
    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    batch = next(
        iter(train_loader)
    )

    print(
        "\nDataLoader batch test:"
    )

    print(
        f"Batch image shape: "
        f"{tuple(batch['image'].shape)}"
    )

    print(
        f"Batch mask shape: "
        f"{tuple(batch['task1_segmentation'].shape)}"
    )

    expected_image_shape = (
        4,
        3,
        image_size,
        image_size,
    )

    expected_mask_shape = (
        4,
        1,
        image_size,
        image_size,
    )

    if tuple(
        batch["image"].shape
    ) != expected_image_shape:
        raise RuntimeError(
            "Unexpected batch image shape. "
            f"Expected {expected_image_shape}, "
            f"received "
            f"{tuple(batch['image'].shape)}."
        )

    if tuple(
        batch[
            "task1_segmentation"
        ].shape
    ) != expected_mask_shape:
        raise RuntimeError(
            "Unexpected batch mask shape. "
            f"Expected {expected_mask_shape}, "
            f"received "
            f"{tuple(batch['task1_segmentation'].shape)}."
        )

    print(
        "\nDataset and DataLoader tests passed."
    )
