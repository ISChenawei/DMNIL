import os
import shutil

"""
This script creates a new dataset by copying images from the original satellite dataset.  
For each subfolder in the original dataset, it copies the first image multiple times  
(renamed sequentially) into the target dataset structure.  

- Original dataset path: DMNIL/dataset/DenseUAV/train/satellite_ini 
- Target dataset path:   DMNIL/dataset/DenseUAV/train/satellite
"""
# data path
source_folder = 'DMNIL/dataset/DenseUAV/train/satellite_ini'

destination_folder = 'DMNIL/dataset/DenseUAV/train/satellite'

os.makedirs(destination_folder, exist_ok=True)


for subfolder in os.listdir(source_folder):
    source_subfolder_path = os.path.join(source_folder, subfolder)
    destination_subfolder_path = os.path.join(destination_folder, subfolder)

    os.makedirs(destination_subfolder_path, exist_ok=True)

    if os.path.isdir(source_subfolder_path):

        image_count = 1


        for image_file in os.listdir(source_subfolder_path):
            source_image_path = os.path.join(source_subfolder_path, image_file)

            for _ in range(2):
                new_image_name = f"{str(image_count).zfill(4)}{os.path.splitext(image_file)[1]}"
                destination_image_path = os.path.join(destination_subfolder_path, new_image_name)

                shutil.copy(source_image_path, destination_image_path)
                image_count += 1

print("新文件夹生成完成！")
