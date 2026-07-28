import torch
import json
import csv
import numpy as np
import cv2
from pathlib import Path
from torch.utils.data import DataLoader, Subset

# Import your architectures and data loaders
from step14_task1_training import build_task1_model
from step17_task2_training import build_task2_model, Task2ModelConfig, ATTRIBUTES
from step12_data_augmentation import LesionDataset, build_val_transform

# ---------------------------------------------------------------------
# 1. Rubric Logic Helpers
# ---------------------------------------------------------------------

def get_border_category(area, perimeter):
    if area == 0:
        return "regular"
    index = (perimeter**2) / (4 * np.pi * area)
    return "irregular" if index >= 1.60 else "regular"

def get_size_category(lesion_area, total_image_area):
    ratio = lesion_area / total_image_area
    if ratio < 0.08:
        return "small"
    elif 0.08 <= ratio <= 0.25:
        return "moderate"
    else: 
        return "large"

def get_attribute_status(p_attr):
    if p_attr >= 0.60:
        return "present"
    elif p_attr <= 0.40:
        return "absent"
    else:
        return "uncertain"

# ---------------------------------------------------------------------
# 2. Main Report Generation Pipeline
# ---------------------------------------------------------------------

def generate_task3_report(sample_mode=False):
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        
    print(f"\nGenerating Task 3 Reports on {device}...")
    
    # Setup Output Directories
    output_dir = Path("outputs/task3_reports")
    json_dir = output_dir / "json"
    json_dir.mkdir(parents=True, exist_ok=True)
    
    csv_path = output_dir / "Summary_reports_text.csv"
    
    # Load Models
    task1_model = build_task1_model().to(device)
    task1_path = Path("outputs/training_results/task1_best_model.pth")
    if task1_path.exists():
        task1_model.load_state_dict(torch.load(task1_path, map_location=device, weights_only=True))
    task1_model.eval()

    config = Task2ModelConfig(use_classification_head=False)
    task2_model = build_task2_model(config).to(device)
    task2_path = Path("outputs/training_results/task2_best_model.pth")
    if task2_path.exists():
        task2_model.load_state_dict(torch.load(task2_path, map_location=device, weights_only=True))
    task2_model.eval()

    # Load Dataset
    image_size = 256 
    total_image_pixels = image_size * image_size
    val_transform = build_val_transform(image_size=image_size)
    
    val_dataset = LesionDataset(fold=0, role="val", transform=val_transform, include_task2=False)

    # Apply the sample filter if requested
    if sample_mode:
        print("--- SAMPLE MODE ON: Processing only the first 10 images ---")
        val_dataset = Subset(val_dataset, range(min(10, len(val_dataset))))
    else:
        print(f"Processing all {len(val_dataset)} images for final submission...")

    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)

    # Open CSV for writing
    with open(csv_path, mode='w', newline='', encoding='utf-8') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['image_id', 'findings']) 

        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                images = batch["image"].to(device)
                
                # Format IDs: raw for CSV, zero-padded for JSON
                raw_id = batch["image_id"][0]
                csv_image_id = str(raw_id)
                json_image_id = str(raw_id).zfill(6) if isinstance(raw_id, int) or str(raw_id).isdigit() else str(raw_id)

                # --- T1: GEOMETRIC CALCULATIONS ---
                t1_logits = task1_model(images)
                t1_mask_np = (torch.sigmoid(t1_logits) > 0.5).squeeze().cpu().numpy().astype(np.uint8)
                
                contours, _ = cv2.findContours(t1_mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                lesion_area_pixels, perimeter_pixels = 0, 0
                if contours:
                    largest_contour = max(contours, key=cv2.contourArea)
                    lesion_area_pixels = int(cv2.contourArea(largest_contour))
                    perimeter_pixels = int(cv2.arcLength(largest_contour, True)) 

                size_cat = get_size_category(lesion_area_pixels, total_image_pixels)
                border_cat = get_border_category(lesion_area_pixels, perimeter_pixels)

                # --- T2: ATTRIBUTE PROBABILITY CALCULATIONS ---
                t2_seg_logits, _ = task2_model(images)
                t2_seg_probs = torch.sigmoid(t2_seg_logits).squeeze().cpu().numpy()
                
                presence_data = {}
                status_dict = {} 
                
                for idx, attr_name in enumerate(ATTRIBUTES):
                    attr_prob_map = t2_seg_probs[idx]
                    
                    if lesion_area_pixels > 0:
                        pixels_inside_roi = attr_prob_map[t1_mask_np == 1]
                        p_attr = float(np.mean(pixels_inside_roi))
                    else:
                        p_attr = 0.0
                    
                    attr_status = get_attribute_status(p_attr)
                    status_dict[attr_name] = attr_status
                    
                    presence_data[attr_name] = {
                        "prob": round(p_attr, 4), 
                        "status": attr_status
                    }

                # --- BUILD JSON ---
                report_json = {
                    "image_id": json_image_id,
                    "split": "test",
                    "model_version": "task1_best_model.pth, task2_best_model.pth",
                    "attributes_order": ATTRIBUTES,
                    "outputs": {
                        "presence": presence_data
                    }
                }

                json_path = json_dir / f"{json_image_id}.json"
                with open(json_path, 'w') as f:
                    json.dump(report_json, f, indent=4)
                
                # --- GENERATE TEXT & CSV LOGGING ---
                report_text = (
                    f"The lesion is {size_cat} with {border_cat} borders. "
                    f"Pigment network is {status_dict['pigment_network']}; "
                    f"Negative network is {status_dict['negative_network']}; "
                    f"Streaks are {status_dict['streaks']}; "
                    f"Milia-like cysts are {status_dict['milia_like_cysts']}; "
                    f"Globules are {status_dict['globules']}."
                )
                
                csv_writer.writerow([csv_image_id, report_text])
                
                # Print progress
                if sample_mode:
                    print(f"Processed {json_image_id}...")
                elif (i + 1) % 50 == 0:
                    print(f"Processed {i + 1}/{len(val_dataset)} images...")

    print(f"\nTask 3 Complete! Check '{output_dir}' for the files.")

# ---------------------------------------------------------------------
# 3. Interactive CLI Menu
# ---------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("TASK 3: ANCHORED FINDINGS REPORT GENERATOR")
    print("=" * 60)
    print("1. Process FULL dataset (For Final Submission)")
    print("2. Process SAMPLE (First 10 images for testing)")
    print("=" * 60)
    
    while True:
        choice = input("Enter your choice (1 or 2): ").strip()
        
        if choice == '1':
            generate_task3_report(sample_mode=False)
            break
        elif choice == '2':
            generate_task3_report(sample_mode=True)
            break
        else:
            print("Invalid input. Please type '1' or '2'.")