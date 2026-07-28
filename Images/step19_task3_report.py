import torch
import json
import numpy as np
import cv2
from pathlib import Path
from torch.utils.data import DataLoader, Subset

# Import your architectures and data loaders
# Note: step17_task2 contains ATTRIBUTES list and Task2Model architecture.
# We will explicitly disable the classification head in config.
from step14_task1_training import build_task1_model
from step17_task2_training import build_task2_model, Task2ModelConfig, ATTRIBUTES
from step12_data_augmentation import LesionDataset, build_val_transform

# ---------------------------------------------------------------------
# 1. Rubric Logic Helpers (based on screenshot 1448-02-13 at 11.58.29 AM)
# ---------------------------------------------------------------------

def get_border_category(area, perimeter):
    """Calculates irregularity index and maps to category string."""
    if area == 0:
        return 0.0, "regular"
    
    # Formula: index = P^2 / (4 * pi * A)
    index = (perimeter**2) / (4 * np.pi * area)
    
    category = "irregular" if index >= 1.60 else "regular"
    return round(float(index), 3), category

def get_size_category(lesion_area, total_image_area):
    """Calculates area ratio and maps to size category."""
    # Formula: ratio = lesion_pixels / total_pixels
    ratio = lesion_area / total_image_area
    
    if ratio < 0.08:
        category = "small"
    elif 0.08 <= ratio <= 0.25:
        category = "moderate"
    else: # ratio > 0.25
        category = "large"
        
    return round(float(ratio), 4), category

def get_attribute_status(p_attr):
    """Maps probability to three-tier status string."""
    if p_attr >= 0.60:
        return "present"
    elif p_attr <= 0.40:
        return "absent"
    else: # 0.40 < p_attr < 0.60
        return "uncertain"

# ---------------------------------------------------------------------
# 2. Main Report Generation Pipeline
# ---------------------------------------------------------------------

def generate_task3_report():
    # 1. Setup Hardware
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        
    print(f"Generating Rubric-Compliant Task 3 Reports on {device}...")
    
    # 2. Output directory
    output_dir = Path("outputs/task3_reports")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 3. Load Task 1 Model (Lesion Segmentation)
    task1_model = build_task1_model().to(device)
    task1_path = Path("outputs/training_results/task1_best_model.pth")
    if task1_path.exists():
        task1_model.load_state_dict(torch.load(task1_path, map_location=device, weights_only=True))
    else:
        print("Warning: Task 1 model not found. Using dummy weights for pipeline test.")
    task1_model.eval()

    # 4. Load Task 2 Model (Attribute Segmentation)
    # Crucial Fix: We explicitly turn OFF the custom classification head
    # as the rubric requires probability calculation from segmentation masks.
    config = Task2ModelConfig(use_classification_head=False)
    task2_model = build_task2_model(config).to(device)
    task2_path = Path("outputs/training_results/task2_best_model.pth")
    if task2_path.exists():
        task2_model.load_state_dict(torch.load(task2_path, map_location=device, weights_only=True))
    else:
        print("Warning: Task 2 model not found. Using dummy weights for pipeline test.")
    task2_model.eval()

    # 5. Load Dataset (Rapid test on first 10 validation images)
    image_size = 256 # Standardized size
    total_image_pixels = image_size * image_size
    val_transform = build_val_transform(image_size=image_size)
    full_dataset = LesionDataset(fold=0, role="val", transform=val_transform, include_task2=False)
    
    # Slice the first 10 so we don't wait for all 540
    val_dataset = Subset(full_dataset, range(10))
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)

    print(f"\nProcessing {len(val_dataset)} images...\n")

    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device)
            image_id = batch["image_id"][0]

            # =========================================================
            # --- T1: GEOMETRIC CALCULATIONS (Size & Border) ---
            # =========================================================
            t1_logits = task1_model(images)
            # Create boolean mask (threshold=0.5)
            t1_mask_np = (torch.sigmoid(t1_logits) > 0.5).squeeze().cpu().numpy().astype(np.uint8)
            
            # Use OpenCV to calculate area and perimeter from mask contours
            contours, _ = cv2.findContours(t1_mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            lesion_area_pixels = 0
            perimeter_pixels = 0
            
            if contours:
                # Find largest blob (assuming largest is the lesion)
                largest_contour = max(contours, key=cv2.contourArea)
                
                lesion_area_pixels = int(cv2.contourArea(largest_contour))
                # arcLength second param 'True' implies closed contour
                perimeter_pixels = int(cv2.arcLength(largest_contour, True)) 

            # Apply Rubric Logic for size and border categories
            area_ratio, size_cat = get_size_category(lesion_area_pixels, total_image_pixels)
            irreg_index, border_cat = get_border_category(lesion_area_pixels, perimeter_pixels)

            # =========================================================
            # --- T2: ATTRIBUTE PROBABILITY CALCULATIONS (ROI) ---
            # =========================================================
            # T2 output is shape [B, 5, H, W] - raw logits. Note: cls_logits is None.
            t2_seg_logits, _ = task2_model(images)
            
            # Fix: Apply Sigmoid immediately to convert raw logits to 0.0 - 1.0 probability maps
            t2_seg_probs = torch.sigmoid(t2_seg_logits).squeeze().cpu().numpy() # shape [5, H, W]
            
            attribute_data = {}
            for idx, attr_name in enumerate(ATTRIBUTES):
                attr_prob_map = t2_seg_probs[idx]
                
                # Rubric Requirement: Calculate mean only inside the lesion ROI boundary.
                # We extract pixels from attribute map where T1 mask is True (== 1)
                if lesion_area_pixels > 0:
                    pixels_inside_roi = attr_prob_map[t1_mask_np == 1]
                    # This is the actual probability used: mean of per-pixel probs inside boundary.
                    p_attr = float(np.mean(pixels_inside_roi))
                else:
                    # Edge case: No lesion detected, probability is 0
                    p_attr = 0.0
                
                # Apply Rubric Thresholds (0.40 / 0.60)
                attr_status = get_attribute_status(p_attr)
                
                attribute_data[attr_name] = {
                    "prob": round(p_attr, 3), # Matching slide decimal places
                    "status": attr_status
                }

            # =========================================================
            # --- BUILD JSON (Matching Schema Output slide) ---
            # =========================================================
            report = {
                "image_id": image_id,
                "metadata": {
                    "split": "val",
                    "resolution": f"{image_size}x{image_size}"
                },
                "structure_analysis": {
                    "overall_lesion": {
                        "size_category": size_cat,
                        "area_ratio": area_ratio,
                        "border_category": border_cat,
                        "irregularity_index": irreg_index
                    },
                    "attributes_map": attribute_data
                }
            }

            # Save the JSON file
            report_path = output_dir / f"report_{image_id}.json"
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=4)
            
            print(f"[{image_id}] size: {size_cat} ({area_ratio:.3f}) | border: {border_cat} | json saved.")

    print(f"\nPipeline complete! 10 rubric-compliant JSON reports saved to '{output_dir}'.")

if __name__ == "__main__":
    # Ensure dependencies from other tasks are visible
    # Make sure your project structure allows imports like: from step12... import ...
    try:
        generate_task3_report()
    except ImportError as e:
        print(f"\nImport Error: {e}")
        print("Ensure step12, step14, and step17 are present in the 'Images' directory ")
        print("or that your pythonpath includes them.\n")
