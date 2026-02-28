"""
remove_outliers.py — 移除异常数据文件（共 6 个严重异常，severity >= 1.0）
运行一次即可：python Remove.py
"""

import os

DATA_DIR = "data"

# 从 0/ 文件夹移除（5 个严重异常文件，表现为 UPDRS=1 特征，很可能是误标）
REMOVE_FROM_0 = [
    "PD-Ruijin_342622197503205933_tappingleft1_2021-01-04_17-32-49.txt",  # +1.97
    "PD-Ruijin_310104195412193651_tappingright1_2021-02-24_13-59-02.txt", # +1.49
    "PD-Ruijin_350322195109020011_tappingright1_2020-11-13_16-40-33.txt", # +1.17
    "PD-Ruijin_340502195305230626_tappingright1_2020-08-31_15-12-35.txt", # +1.09
    "PD-Ruijin_330724195105251817_tappingright1_2020-11-03_18-48-48.txt"  # +1.01
]

# 从 1/ 文件夹移除（1 个严重异常文件，表现为 UPDRS=0 特征）
REMOVE_FROM_1 = [
    "PD-Ruijin_362426196205031828_tappingright1_2020-09-22_15-44-54.txt"  # +1.57
]

def remove_files(folder_name, file_list):
    folder_path = os.path.join(DATA_DIR, folder_name)
    if not os.path.exists(folder_path):
        print(f"文件夹不存在: {folder_path}")
        return

    removed_count = 0
    for filename in file_list:
        filepath = os.path.join(folder_path, filename)
        if os.path.exists(filepath):
            os.remove(filepath)
            print(f"已删除: {filepath}")
            removed_count += 1
        else:
            print(f"未找到 (可能已删除): {filepath}")

    print(f"[{folder_name}/] 清理完成，共删除 {removed_count} 个文件。\n")

if __name__ == "__main__":
    print("开始清理严重异常数据...")
    remove_files("0", REMOVE_FROM_0)
    remove_files("1", REMOVE_FROM_1)
    print("清理任务全部结束！")