import os
import shutil

"""
This script creates a new dataset by copying images from the original satellite dataset.  
For each subfolder in the original dataset, it copies the first image multiple times  
(renamed sequentially) into the target dataset structure.  

- Original dataset path: DMNIL/dataset/SUES-200/train/150/satellite_origin  
- Target dataset path:   DMNIL/dataset/SUES-200/150/train/satellite  
"""

# data path
original_dataset_root = "DMNIL/dataset/SUES-200/train/150/satellite_origin"  # original dataset path
target_dataset_root = "DMNIL/dataset/SUES-200/150/train/satellite"      # target dataset path

# create the target root directory if it does not exist
if not os.path.exists(target_dataset_root):
    os.makedirs(target_dataset_root)

# iterate through all subfolders inside the satellite folder
for folder_name in os.listdir(original_dataset_root):
    folder_path = os.path.join(original_dataset_root, folder_name)

    # ensure we only process directories
    if os.path.isdir(folder_path):
        # create the corresponding subfolder in the target directory
        new_folder_path = os.path.join(target_dataset_root, folder_name)
        os.makedirs(new_folder_path, exist_ok=True)

        # get all image files in the subfolder
        image_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

        # if images exist, copy the first one multiple times (renamed)
        if image_files:
            for i in range(0, 49):
                # copy the first image in the folder
                source_image_path = os.path.join(folder_path, image_files[0])
                new_image_name = f"{i:02d}.jpg"  # naming: 00.jpg to 49.jpg
                new_image_path = os.path.join(new_folder_path, new_image_name)
                shutil.copy(source_image_path, new_image_path)

print("New dataset has been generated!")