import os
import shutil
import os
import shutil

source_dir = "DMNIL/dataset/DenseUAV/test/gallery_satellite"  # 原始目录路径

target_dirs = {
    "H80": "DMNIL/dataset/DenseUAV/test/gallery_satellite_Height/H80",  # 存储 H80 文件的目录
    "H90": "DMNIL/dataset/DenseUAV/test/gallery_satellite_Height/H90",  # 存储 H90 文件的目录
    "H100": "DMNIL/dataset/DenseUAV/test/gallery_satellite_Height/H100",  # 存储 H100 文件的目录
}

for target_dir in target_dirs.values():
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

for subdir in os.listdir(source_dir):
    subdir_path = os.path.join(source_dir, subdir)
    if os.path.isdir(subdir_path): 
        for file in os.listdir(subdir_path):
            if file.endswith(".tif"): 
                if "_old" in file:
                    prefix = file.split("_old")[0]
                else:
                    prefix = file.split(".")[0]

                if prefix in target_dirs: 
                    target_subdir = os.path.join(target_dirs[prefix], subdir)
                    if not os.path.exists(target_subdir):
                        os.makedirs(target_subdir)

                    src_file = os.path.join(subdir_path, file)
                    dest_file = os.path.join(target_subdir, file)
                    shutil.copy(src_file, dest_file) 
print("文件已按前缀归类，并保存到新的大文件夹中。")


# import os
# import shutil
#
# source_dir = "DMNIL/dataset/DenseUAV/test/query_drone"  # 原始目录路径
#
# target_dirs = {
#     "H80": "DMNIL/dataset/DenseUAV/test/query_drone_Height/H80",  # 存储 H80 文件的目录
#     "H90": "DMNIL/dataset/DenseUAV/test/query_drone_Height/H90",  # 存储 H90 文件的目录
#     "H100": "DMNIL/dataset/DenseUAV/test/query_drone_Height/H100",  # 存储 H100 文件的目录
# }
#
# for target_dir in target_dirs.values():
#     if not os.path.exists(target_dir):
#         os.makedirs(target_dir)
#
# for subdir in os.listdir(source_dir):
#     subdir_path = os.path.join(source_dir, subdir)
#     if os.path.isdir(subdir_path): 
#         for file in os.listdir(subdir_path):
#             if file.endswith(".JPG"): 
#                 if "_old" in file:
#                     prefix = file.split("_old")[0]
#                 else:
#                     prefix = file.split(".")[0]
#
#                 if prefix in target_dirs: 
#                     target_subdir = os.path.join(target_dirs[prefix], subdir)
#                     if not os.path.exists(target_subdir):
#                         os.makedirs(target_subdir)
#
#                     src_file = os.path.join(subdir_path, file)
#                     dest_file = os.path.join(target_subdir, file)
#                     shutil.copy(src_file, dest_file)  # 使用 shutil.copy 保留源文件
#
# print("文件已按前缀归类，并保存到新的大文件夹中。")
