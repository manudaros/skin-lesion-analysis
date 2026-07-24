import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.spatial.distance import directed_hausdorff
from torch.utils.data import DataLoader, Subset
import random

# Import your model blueprint and dataset
from step14_task1_training import build_task1_model
from step12_data_augmentation import LesionDataset, build_val_transform

# ---------------------------------------------------------------------
# 1. Evaluation Metrics
# ---------------------------------------------------------------------
def calculate_metrics(probs, targets, threshold=0.5):
    preds = (probs > threshold).float()
    preds_flat = preds.view(-1)
    targets_flat = targets.view(-1)
    
    intersection = (preds_flat * targets_flat).sum().item()
    union = preds_flat.sum().item() + targets_flat.sum().item() - intersection
    
    dice = (2. * intersection) / (preds_flat.sum().item() + targets_flat.sum().item() + 1e-6)
    iou = intersection / (union + 1e-6)
    
    preds_np = preds.squeeze().cpu().numpy()
    targets_np = targets.squeeze().cpu().numpy()
    
    preds_coords = np.argwhere(preds_np == 1)
    targets_coords = np.argwhere(targets_np == 1)
    
    if len(preds_coords) == 0 or len(targets_coords) == 0:
        hd95 = float('nan')
    else:
        hd_forward = directed_hausdorff(preds_coords, targets_coords)[0]
        hd_backward = directed_hausdorff(targets_coords, preds_coords)[0]
        hd95 = max(hd_forward, hd_backward)

    return dice, iou, hd95

# ---------------------------------------------------------------------
# 2. Visual Generator (New Clean Color Scheme)
# ---------------------------------------------------------------------
def save_evaluation_visual(image, mask_gt, mask_pred, save_path, image_id=""):
    image_np = image.permute(1, 2, 0).cpu().numpy()
    mask_gt_np = mask_gt.squeeze().cpu().numpy()
    mask_pred_np = mask_pred.squeeze().cpu().numpy()
    
    # Normalize image for viewing
    image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min())
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    axes[0].imshow(image_np)
    axes[0].set_title(f"Input: {image_id}")
    
    axes[1].imshow(mask_gt_np, cmap='gray')
    axes[1].set_title("Ground Truth")
    
    axes[2].imshow(mask_pred_np, cmap='gray')
    axes[2].set_title("Prediction")
    
    # 4th Panel: Pure Mask Comparison (No skin image underneath)
    # Background defaults to Black [0, 0, 0]
    error_map = np.zeros((image_np.shape[0], image_np.shape[1], 3))
    
    # True Positive (Correct spots): White
    error_map[(mask_pred_np == 1) & (mask_gt_np == 1)] = [1.0, 1.0, 1.0] 
    
    # False Positive (Extras): Purple
    error_map[(mask_pred_np == 1) & (mask_gt_np == 0)] = [0.6, 0.0, 0.8] 
    
    # False Negative (Missed spots): Red
    error_map[(mask_pred_np == 0) & (mask_gt_np == 1)] = [1.0, 0.0, 0.0] 
    
    axes[3].imshow(error_map)
    axes[3].set_title("Error Map (W: Correct, P: Extra, R: Missed)")
    
    for ax in axes:
        ax.axis('off')
        
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()

# ---------------------------------------------------------------------
# 3. Main Evaluation Loop
# ---------------------------------------------------------------------
def evaluate_model(eval_mode):
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        
    print(f"\nEvaluating on {device}...")
    
    output_dir = Path("outputs/evaluation_visuals")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Make sure this matches whatever size your model was just trained on (128 or 256)
    image_size = 128  
    val_transform = build_val_transform(image_size=image_size)
    full_val_dataset = LesionDataset(fold=0, role="val", transform=val_transform, include_task2=False)
    
    # Apply user selection
    if eval_mode == "1":
        print("Mode 1: Slicing 10 random images...")
        random_indices = random.sample(range(len(full_val_dataset)), 10)
        val_dataset = Subset(full_val_dataset, random_indices)
        max_images_to_save = 10
    else:
        print(f"Mode 2: Loading entire validation set ({len(full_val_dataset)} images)...")
        val_dataset = full_val_dataset
        max_images_to_save = 10 # Still capping at 10 visuals so we don't spam your hard drive
        
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)
    
    model = build_task1_model().to(device)
    model_path = Path("outputs/training_results/task1_best_model.pth")
    
    if not model_path.exists():
        raise FileNotFoundError(f"Could not find model at {model_path}.")
        
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    total_dice, total_iou, total_hd95 = 0.0, 0.0, 0.0
    valid_hd_count = 0
    images_saved = 0
    
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            images = batch["image"].to(device)
            masks = batch["task1_segmentation"].to(device)
            image_id = batch["image_id"][0]
            
            logits = model(images)
            probs = torch.sigmoid(logits)
            
            dice, iou, hd95 = calculate_metrics(probs, masks)
            total_dice += dice
            total_iou += iou
            if not np.isnan(hd95):
                total_hd95 += hd95
                valid_hd_count += 1
            
            if images_saved < max_images_to_save:
                save_path = output_dir / f"val_result_{image_id}.png"
                save_evaluation_visual(images[0], masks[0], (probs[0] > 0.5).float(), save_path, image_id)
                images_saved += 1
                
    n_val = len(val_loader)
    print("\n" + "="*50)
    print("FINAL VALIDATION METRICS")
    print("="*50)
    print(f"Evaluated {n_val} images.")
    print(f"Average Dice Score: {total_dice/n_val:.4f}")
    print(f"Average IoU (Jaccard): {total_iou/n_val:.4f}")
    if valid_hd_count > 0:
        print(f"Average 95% Hausdorff: {total_hd95/valid_hd_count:.2f}")
    print("="*50)
    print(f"Check the '{output_dir}' folder for your generated error maps!")

if __name__ == "__main__":
    print("\n--- EVALUATION MENU ---")
    print("1) Fast Mode: Evaluate and visualize 10 random images")
    print("2) Full Mode: Evaluate the entire validation set (visualizes first 10)")
    
    choice = input("Select an option (1 or 2): ").strip()
    
    if choice in ["1", "2"]:
        evaluate_model(eval_mode=choice)
    else:
        print("Invalid choice. Please run the script again and select 1 or 2.")