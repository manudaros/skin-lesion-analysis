import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

# Import your architectures and data loaders
from step14_task1_training import build_task1_model
from step17_task2_training import build_task2_model, Task2ModelConfig, ATTRIBUTES
from step12_data_augmentation import LesionDataset, build_val_transform

# ---------------------------------------------------------------------
# 1. Helper Functions
# ---------------------------------------------------------------------

def unnormalize(tensor):
    """Reverts ImageNet normalization so the image looks normal to the human eye."""
    mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)
    img = tensor.permute(1, 2, 0).cpu().numpy()
    img = std * img + mean
    return np.clip(img, 0, 1)

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
# 2. Main Visualization Function
# ---------------------------------------------------------------------

def visualize_pipeline(image_index=0):
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        
    print(f"\nLoading models on {device}...")
    
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

    # Handle Random Selection
    if image_index == 'random':
        image_index = np.random.randint(0, len(val_dataset))
        print(f"Randomly selected index: {image_index}")

    # Validate Index
    if image_index >= len(val_dataset) or image_index < 0:
        print(f"Error: Index {image_index} is out of bounds. Max index is {len(val_dataset)-1}.")
        return

    # Get Single Sample
    sample = val_dataset[image_index]
    image_tensor = sample["image"].unsqueeze(0).to(device) # Add batch dimension
    image_id = sample["image_id"]
    
    print(f"Visualizing Pipeline for Image ID: {image_id}")

    with torch.no_grad():
        # --- Task 1: Lesion Segmentation ---
        t1_logits = task1_model(image_tensor)
        t1_prob = torch.sigmoid(t1_logits).squeeze().cpu().numpy()
        t1_mask = (t1_prob > 0.5).astype(np.uint8)
        
        # Calculate Geometric features
        contours, _ = cv2.findContours(t1_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        lesion_area_pixels, perimeter_pixels = 0, 0
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            lesion_area_pixels = int(cv2.contourArea(largest_contour))
            perimeter_pixels = int(cv2.arcLength(largest_contour, True)) 

        size_cat = get_size_category(lesion_area_pixels, total_image_pixels)
        border_cat = get_border_category(lesion_area_pixels, perimeter_pixels)

        # --- Task 2: Attribute Detection ---
        t2_logits, _ = task2_model(image_tensor)
        t2_probs = torch.sigmoid(t2_logits).squeeze().cpu().numpy()
        
        status_dict = {}
        for idx, attr_name in enumerate(ATTRIBUTES):
            attr_prob_map = t2_probs[idx]
            
            if lesion_area_pixels > 0:
                p_attr = float(np.mean(attr_prob_map[t1_mask == 1]))
            else:
                p_attr = 0.0
                
            status_dict[attr_name] = get_attribute_status(p_attr)

        # --- Task 3: Generated Text ---
        report_text = (
            f"The lesion is {size_cat} with {border_cat} borders. "
            f"Pigment network is {status_dict['pigment_network']}; "
            f"Negative network is {status_dict['negative_network']}; "
            f"Streaks are {status_dict['streaks']}; "
            f"Milia-like cysts are {status_dict['milia_like_cysts']}; "
            f"Globules are {status_dict['globules']}."
        )

    # ---------------------------------------------------------------------
    # 3. Matplotlib Dashboard
    # ---------------------------------------------------------------------
    
    rgb_image = unnormalize(sample["image"])
    
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(f"End-to-End Pipeline Output (Image: {image_id} | Index: {image_index})", fontsize=18, fontweight='bold')

    # Row 1: Original Image & Task 1
    ax1 = plt.subplot2grid((3, 5), (0, 1), colspan=1)
    ax1.imshow(rgb_image)
    ax1.set_title("Original Image")
    ax1.axis("off")

    ax2 = plt.subplot2grid((3, 5), (0, 2), colspan=1)
    ax2.imshow(t1_mask, cmap="gray")
    ax2.set_title("Task 1: Lesion Mask")
    ax2.axis("off")
    
    # Create an overlay
    overlay = rgb_image.copy()
    overlay[t1_mask == 1] = overlay[t1_mask == 1] * 0.5 + np.array([1, 0, 0]) * 0.5 # Red tint
    
    ax3 = plt.subplot2grid((3, 5), (0, 3), colspan=1)
    ax3.imshow(overlay)
    ax3.set_title(f"Task 1: Overlay\nSize: {size_cat} | Border: {border_cat}")
    ax3.axis("off")

    # Row 2: Task 2 Attributes
    for i, attr_name in enumerate(ATTRIBUTES):
        ax = plt.subplot2grid((3, 5), (1, i))
        # Showing the probability heatmap masked by the lesion ROI
        heatmap = t2_probs[i].copy()
        heatmap[t1_mask == 0] = 0 # Blank out outside ROI to show focus
        
        im = ax.imshow(heatmap, cmap="magma", vmin=0, vmax=1)
        formatted_name = attr_name.replace("_", " ").title()
        status = status_dict[attr_name].upper()
        
        ax.set_title(f"{formatted_name}\n({status})", fontsize=10)
        ax.axis("off")

    # Row 3: Task 3 Text Report
    ax_text = plt.subplot2grid((3, 5), (2, 0), colspan=5)
    ax_text.axis("off")
    
    # Add textbox
    bbox_props = dict(boxstyle="round,pad=1", fc="#f4f4f4", ec="black", lw=2)
    ax_text.text(0.5, 0.5, f"Task 3 Generated Report:\n\n{report_text}", 
                 fontsize=14, ha="center", va="center", bbox=bbox_props, wrap=True)

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()

# ---------------------------------------------------------------------
# 4. Interactive CLI Menu
# ---------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("TASK 1-3: FULL PIPELINE VISUALIZER")
    print("=" * 60)
    print("1. Visualize a RANDOM image")
    print("2. Visualize a SPECIFIC image by index number")
    print("=" * 60)
    
    while True:
        choice = input("Enter your choice (1 or 2): ").strip()
        
        if choice == '1':
            visualize_pipeline(image_index='random')
            break
        elif choice == '2':
            idx_str = input("Enter the image index (e.g., 0, 42, 100): ").strip()
            try:
                idx = int(idx_str)
                visualize_pipeline(image_index=idx)
                break
            except ValueError:
                print("Invalid input. Please enter a whole number.")
        else:
            print("Invalid input. Please type '1' or '2'.")