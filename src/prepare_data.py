import os
import math
import shutil
import pandas as pd
import numpy as np

def create_yolo_dataset(csv_path, images_dir, output_dir="dataset"):
    print("Setting up YOLO dataset structure...")
    df = pd.read_csv(csv_path)
    
    # Create folder structure
    for split in ['train', 'val']:
        os.makedirs(os.path.join(output_dir, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'labels', split), exist_ok=True)
        
    unique_images = df['image'].unique()
    np.random.shuffle(unique_images)
    
    # 80% Train, 20% Validation split
    split_idx = int(len(unique_images) * 0.8)
    train_imgs = unique_images[:split_idx]
    
    for img_name in unique_images:
        split = 'train' if img_name in train_imgs else 'val'
        img_data = df[df['image'] == img_name]
        
        # Copy image to new folder
        src_img_path = os.path.join(images_dir, img_name)
        dst_img_path = os.path.join(output_dir, 'images', split, img_name)
        if os.path.exists(src_img_path):
            shutil.copy(src_img_path, dst_img_path)
        
        # Create label file
        label_path = os.path.join(output_dir, 'labels', split, img_name.replace('.png', '.txt'))
        
        with open(label_path, 'w') as f:
            for _, row in img_data.iterrows():
                # Normalize bounding box (0 to 1 scale)
                img_w, img_h = 640, 480
                nx_center = (row['bbox_x'] + row['bbox_w'] / 2) / img_w
                ny_center = (row['bbox_y'] + row['bbox_h'] / 2) / img_h
                nw = row['bbox_w'] / img_w
                nh = row['bbox_h'] / img_h
                
                # Normalize Center Keypoint
                k1_x = row['center_x'] / img_w
                k1_y = row['center_y'] / img_h
                
                # Calculate Tab Keypoint using angle (assuming radius of ~15 pixels for the tab)
                angle_rad = math.radians(row['angle_deg'])
                # Y goes down in images, so we subtract sin
                tab_x = row['center_x'] + 15 * math.cos(angle_rad)
                tab_y = row['center_y'] - 15 * math.sin(angle_rad)
                k2_x = tab_x / img_w
                k2_y = tab_y / img_h
                
                # YOLO Pose Format: class x_center y_center width height px1 py1 p1_vis px2 py2 p2_vis
                # Visibility = 2 (visible and labeled)
                line = f"0 {nx_center:.5f} {ny_center:.5f} {nw:.5f} {nh:.5f} {k1_x:.5f} {k1_y:.5f} 2 {k2_x:.5f} {k2_y:.5f} 2\n"
                f.write(line)
                
    print(f"Dataset ready! {len(train_imgs)} train images, {len(unique_images)-len(train_imgs)} validation images.")

if __name__ == "__main__":
    # Ensure you change 'path/to/images' to where your downloaded images are
    create_yolo_dataset("annotations.csv", "images/")
