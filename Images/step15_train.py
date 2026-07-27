import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt, binary_erosion
from torch.utils.data import DataLoader

from step14_task1_training import build_task1_model, Task1ModelConfig
from step12_data_augmentation import (
    LesionDataset,
    build_train_transform,
    build_val_transform,
)


# ---------------------------------------------------------------------
# 0. Device selection
# ---------------------------------------------------------------------

def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------
# 1. Combined Loss (BCE + Dice)
# ---------------------------------------------------------------------

class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets, smooth=1e-6):
        bce_loss = self.bce(logits, targets)
        probs = torch.sigmoid(logits)
        intersection = (probs * targets).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
        dice_score = (2.0 * intersection + smooth) / (union + smooth)
        dice_loss = 1.0 - dice_score.mean()
        return (self.bce_weight * bce_loss) + (self.dice_weight * dice_loss)


# ---------------------------------------------------------------------
# 2. Metrics (per single image)
# ---------------------------------------------------------------------

def _boundary(mask_bool: np.ndarray) -> np.ndarray:
    if mask_bool.sum() == 0:
        return mask_bool
    return mask_bool & ~binary_erosion(mask_bool)


def hausdorff_95_from_masks(pred_bool, target_bool) -> float:
    if pred_bool.sum() == 0 or target_bool.sum() == 0:
        return float("nan")
    pred_b = _boundary(pred_bool)
    target_b = _boundary(target_bool)
    dt_to_target = distance_transform_edt(~target_b)
    dt_to_pred = distance_transform_edt(~pred_b)
    d1 = dt_to_target[pred_b]
    d2 = dt_to_pred[target_b]
    return float(np.percentile(np.concatenate([d1, d2]), 95))


def calculate_metrics_single(prob, target, threshold=0.5, compute_hd95=True):
    pred = (prob > threshold).float()
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)

    intersection = (pred_flat * target_flat).sum().item()
    pred_sum = pred_flat.sum().item()
    target_sum = target_flat.sum().item()
    union = pred_sum + target_sum - intersection

    dice = (2.0 * intersection) / (pred_sum + target_sum + 1e-6)
    iou = intersection / (union + 1e-6)

    if not compute_hd95:
        return dice, iou, float("nan")

    pred_np = pred.squeeze().cpu().numpy().astype(bool)
    target_np = target.squeeze().cpu().numpy().astype(bool)
    return dice, iou, hausdorff_95_from_masks(pred_np, target_np)


# ---------------------------------------------------------------------
# 3. Error map visualization
# ---------------------------------------------------------------------

def save_error_map(image, mask_gt, mask_pred, save_path):
    image_np = image.permute(1, 2, 0).cpu().numpy()
    mask_gt_np = mask_gt.squeeze().cpu().numpy()
    mask_pred_np = mask_pred.squeeze().cpu().numpy()

    denom = image_np.max() - image_np.min()
    if denom > 0:
        image_np = (image_np - image_np.min()) / denom

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(image_np)
    axes[0].set_title("Input Image")
    axes[1].imshow(mask_gt_np, cmap='gray')
    axes[1].set_title("Ground Truth")
    axes[2].imshow(mask_pred_np, cmap='gray')
    axes[2].set_title("Prediction")

    error_map = np.zeros((*mask_gt_np.shape, 3), dtype=np.float32)
    error_map[(mask_pred_np == 1) & (mask_gt_np == 1)] = [0, 1, 0]
    error_map[(mask_pred_np == 1) & (mask_gt_np == 0)] = [1, 0, 0]
    error_map[(mask_pred_np == 0) & (mask_gt_np == 1)] = [0, 0, 1]

    axes[3].imshow(image_np)
    axes[3].imshow(error_map, alpha=0.5)
    axes[3].set_title("Error Map Overlay")
    for ax in axes:
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# ---------------------------------------------------------------------
# 4. Training loop
# ---------------------------------------------------------------------

def train_model(train_loader, val_loader, epochs=25, lr=1e-4,
                compute_hd95=True):
    device = select_device()
    print(f"Training on {device}...")

    model = build_task1_model().to(device)
    criterion = BCEDiceLoss(bce_weight=0.5, dice_weight=0.5)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Halve the learning rate when validation Dice stops improving.
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )

    output_dir = Path("outputs/training_results")
    output_dir.mkdir(parents=True, exist_ok=True)

    best_dice = -1.0

    for epoch in range(epochs):
        # ---- TRAIN ----
        model.train()
        train_loss = 0.0
        t0 = time.time()

        for step, batch in enumerate(train_loader):
            images = batch["image"].to(device)
            masks = batch["task1_segmentation"].to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            if step % 20 == 0:
                elapsed = time.time() - t0
                print(f"  epoch {epoch+1} step {step}/{len(train_loader)} "
                      f"loss {loss.item():.4f}  ({elapsed:.1f}s)")

        # ---- VALIDATE ----
        model.eval()
        val_loss = 0.0
        dice_scores, iou_scores, hd95_scores = [], [], []
        tv = time.time()

        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                images = batch["image"].to(device)
                masks = batch["task1_segmentation"].to(device)
                logits = model(images)
                probs = torch.sigmoid(logits)
                val_loss += criterion(logits, masks).item()

                for b in range(images.size(0)):
                    dice, iou, hd95 = calculate_metrics_single(
                        probs[b:b+1], masks[b:b+1], compute_hd95=compute_hd95)
                    dice_scores.append(dice)
                    iou_scores.append(iou)
                    if not np.isnan(hd95):
                        hd95_scores.append(hd95)

                if i == 0:
                    save_error_map(images[0], masks[0],
                                   (probs[0] > 0.5).float(),
                                   output_dir / f"epoch_{epoch}_visual.png")

        mean_train_loss = train_loss / len(train_loader)
        mean_val_loss = val_loss / len(val_loader)
        mean_dice = float(np.mean(dice_scores))
        mean_iou = float(np.mean(iou_scores))
        mean_hd95 = float(np.mean(hd95_scores)
                          ) if hd95_scores else float("nan")

        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {mean_train_loss:.4f} | "
              f"Val Loss: {mean_val_loss:.4f} | "
              f"Val Dice: {mean_dice:.4f} | "
              f"Val IoU: {mean_iou:.4f} | "
              f"Val HD95: {mean_hd95:.2f} | "
              f"val {time.time()-tv:.1f}s")

        # Step the scheduler on validation Dice and report the LR.
        scheduler.step(mean_dice)
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"  learning rate: {current_lr:.2e}")

        if mean_dice > best_dice:
            best_dice = mean_dice
            torch.save(model.state_dict(),
                       output_dir / "task1_best_model.pth")
            print(f"  New best model saved (Dice {best_dice:.4f})")

    torch.save(model.state_dict(), output_dir / "task1_last_model.pth")
    print(f"\nTraining complete. Best validation Dice: {best_dice:.4f}")
    return model


# ---------------------------------------------------------------------
# 5. Entry point
# ---------------------------------------------------------------------

if __name__ == "__main__":
    # Relax the MPS memory ceiling that can cause stalls on Apple Silicon.
    os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

    image_size = 384
    batch_size = 8
    epochs = 25
    dev_fold = 0
    COMPUTE_HD95 = True

    print("Building datasets and dataloaders...")
    train_transform = build_train_transform(image_size=image_size)
    val_transform = build_val_transform(image_size=image_size)

    train_dataset = LesionDataset(
        fold=dev_fold, role="train",
        transform=train_transform, include_task2=False)
    val_dataset = LesionDataset(
        fold=dev_fold, role="val",
        transform=val_transform, include_task2=False)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    train_model(train_loader, val_loader, epochs=epochs,
                compute_hd95=COMPUTE_HD95)
