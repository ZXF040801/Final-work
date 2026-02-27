import os
import shutil
import pandas as pd


def copy_files_by_updrs_score(excel_file_path, data_folder_path):
    # 1. 读取 Excel 文件中名为 'Clinical notes' 的工作表
    try:
        # 如果您的 sheet 名称不是 'Clinical notes'，请修改 sheet_name 参数
        # 也可以使用 sheet_name=0 来读取第一个工作表
        df = pd.read_excel(excel_file_path, sheet_name='Clinical notes', engine='openpyxl')
        print("成功读取 Excel 文件。")
    except FileNotFoundError:
        print(f"找不到文件: {excel_file_path}")
        return
    except ValueError as e:
        print(f"读取工作表时出错，请检查工作表名称是否正确: {e}")
        return

    # 检查Excel中是否包含需要的列
    if 'FT Clinical UPDRS Score' not in df.columns or 'Data Filename' not in df.columns:
        print("Excel 文件中缺少必要的列: 'FT Clinical UPDRS Score' 或 'Data Filename'")
        print(f"当前存在的列为: {df.columns.tolist()}")
        return

    # 2. 筛选出得分为 0 或 1 的数据
    filtered_df = df[df['FT Clinical UPDRS Score'].isin([0, 1])]

    # 3. 在当前目录下创建分别叫 0 和 1 的文件夹
    for score in [0, 1]:
        folder_path = str(score)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
            print(f"已创建文件夹: {folder_path}")

    # 4. 根据筛选出的数据，从 data 文件夹复制文件
    success_count = 0
    missing_count = 0

    for index, row in filtered_df.iterrows():
        score = int(row['FT Clinical UPDRS Score'])
        filename = str(row['Data Filename']).strip()

        # 忽略空的或无效的文件名
        if pd.isna(row['Data Filename']) or filename == 'nan' or not filename:
            continue

        # 构建原文件路径和目标文件路径
        src_path = os.path.join(data_folder_path, filename)
        dst_path = os.path.join(str(score), filename)

        # 检查原文件是否存在
        if os.path.exists(src_path):
            try:
                # 复制文件到对应分数的文件夹
                shutil.copy2(src_path, dst_path)
                success_count += 1
                print(f"成功复制: {filename} -> {score}/")
            except Exception as e:
                print(f"复制文件 {filename} 时发生错误: {e}")
        else:
            missing_count += 1
            print(f"文件不存在，跳过: {src_path}")

    print("-" * 30)
    print(f"操作完成！成功复制文件数量: {success_count}，未找到文件数量: {missing_count}")


if __name__ == "__main__":
    # 您的 Excel 文件名
    excel_file = "All-Ruijin-labels.xlsx"

    # 存放所有原始测量数据文件的文件夹名称
    data_folder = "data"

    copy_files_by_updrs_score(excel_file, data_folder)