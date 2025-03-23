import pandas as pd
import os

def split_csv(file_path, output_dir, rows_per_file=10000):  # 調小每個檔案的行數
    os.makedirs(output_dir, exist_ok=True)

    chunk_iterator = pd.read_csv(file_path, chunksize=rows_per_file, encoding="utf-8", on_bad_lines='skip')
    file_number = 1

    for chunk in chunk_iterator:
        new_file_name = os.path.join(output_dir, f'data{file_number}.csv')
        chunk.to_csv(new_file_name, index=False, encoding="utf-8-sig")  
        print(f'已儲存 {new_file_name}')
        file_number += 1

# 測試
file_path = r'C:\Users\yulun\Downloads\38_Training_Data_Set\38_Training_Data_Set\training.csv'
output_dir = r'C:\Users\yulun\Downloads\38_Training_Data_Set\Split_Files'

split_csv(file_path, output_dir, rows_per_file=10000)  # 每個檔案 1 萬行
