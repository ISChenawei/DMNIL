import os
import shutil

# 设置源目录和目标目录
import os
import shutil

# 设置源目录和目标目录
source_dir = "DMNIL/dataset/DenseUAV/test/gallery_satellite"  # 原始目录路径

target_dirs = {
    "H80": "DMNIL/dataset/DenseUAV/test/gallery_satellite_Height/H80",  # 存储 H80 文件的目录
    "H90": "DMNIL/dataset/DenseUAV/test/gallery_satellite_Height/H90",  # 存储 H90 文件的目录
    "H100": "DMNIL/dataset/DenseUAV/test/gallery_satellite_Height/H100",  # 存储 H100 文件的目录
}

# 确保目标目录存在
for target_dir in target_dirs.values():
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

# 遍历原始文件夹中的编号文件夹
for subdir in os.listdir(source_dir):
    subdir_path = os.path.join(source_dir, subdir)
    if os.path.isdir(subdir_path):  # 确保是文件夹
        for file in os.listdir(subdir_path):
            if file.endswith(".tif"):  # 筛选出 .tif 文件
                # 提取文件前缀，包括处理 _old 的情况
                if "_old" in file:
                    prefix = file.split("_old")[0]
                else:
                    prefix = file.split(".")[0]

                if prefix in target_dirs:  # 检查前缀是否在目标目录中
                    # 构造目标文件夹中的对应编号文件夹
                    target_subdir = os.path.join(target_dirs[prefix], subdir)
                    if not os.path.exists(target_subdir):
                        os.makedirs(target_subdir)

                    # 复制文件到目标文件夹中
                    src_file = os.path.join(subdir_path, file)
                    dest_file = os.path.join(target_subdir, file)
                    shutil.copy(src_file, dest_file)  # 使用 shutil.copy 保留源文件

print("文件已按前缀归类，并保存到新的大文件夹中。")


# import os
# import shutil
#
# # 设置源目录和目标目录
# source_dir = "DMNIL/dataset/DenseUAV/test/query_drone"  # 原始目录路径
#
# target_dirs = {
#     "H80": "DMNIL/dataset/DenseUAV/test/query_drone_Height/H80",  # 存储 H80 文件的目录
#     "H90": "DMNIL/dataset/DenseUAV/test/query_drone_Height/H90",  # 存储 H90 文件的目录
#     "H100": "DMNIL/dataset/DenseUAV/test/query_drone_Height/H100",  # 存储 H100 文件的目录
# }
#
# # 确保目标目录存在
# for target_dir in target_dirs.values():
#     if not os.path.exists(target_dir):
#         os.makedirs(target_dir)
#
# # 遍历原始文件夹中的编号文件夹
# for subdir in os.listdir(source_dir):
#     subdir_path = os.path.join(source_dir, subdir)
#     if os.path.isdir(subdir_path):  # 确保是文件夹
#         for file in os.listdir(subdir_path):
#             if file.endswith(".JPG"):  # 筛选出 .tif 文件
#                 # 提取文件前缀，包括处理 _old 的情况
#                 if "_old" in file:
#                     prefix = file.split("_old")[0]
#                 else:
#                     prefix = file.split(".")[0]
#
#                 if prefix in target_dirs:  # 检查前缀是否在目标目录中
#                     # 构造目标文件夹中的对应编号文件夹
#                     target_subdir = os.path.join(target_dirs[prefix], subdir)
#                     if not os.path.exists(target_subdir):
#                         os.makedirs(target_subdir)
#
#                     # 复制文件到目标文件夹中
#                     src_file = os.path.join(subdir_path, file)
#                     dest_file = os.path.join(target_subdir, file)
#                     shutil.copy(src_file, dest_file)  # 使用 shutil.copy 保留源文件
#
# print("文件已按前缀归类，并保存到新的大文件夹中。")