import pandas as pd
import shutil
import os

def classify_frames():
    """
    将SMIC提取的帧按情感分类，并重新映射标签以匹配CASME II格式
    """
    # 读取SMIC数据集的标注文件
    df = pd.read_csv("../../Dataset/SMIC_all_cropped/smic_apex.csv")
    
    # 源文件夹和目标文件夹
    source_base = "../../Dataset/SMIC_onset_apex_offset/"
    out_dir_path = "../../Dataset/SMIC_retinaface_classify/"
    
    # 如果输出文件夹已存在，先删除
    if os.path.exists(out_dir_path):
        shutil.rmtree(out_dir_path)
        print(f"删除已存在的输出文件夹: {out_dir_path}")
    
    # 创建输出目录
    os.makedirs(out_dir_path, exist_ok=True)
    
    # 标签映射：SMIC -> CASME II 3类格式
    # SMIC: 0(negative), 1(positive), 2(surprise)
    # CASME II 3类: 0(positive), 1(negative), 2(surprise)
    label_mapping = {
        0: 1,  # SMIC negative -> CASME II negative
        1: 0,  # SMIC positive -> CASME II positive
        2: 2   # SMIC surprise -> CASME II surprise
    }
    
    # 处理每一行数据
    for index, row in df.iterrows():
        subject_id = f"s{row['subject']}"
        clip_name = row['clip']
        
        # 获取映射后的标签
        original_label = row['label']
        mapped_label = label_mapping[original_label]
        
        # 创建目标目录
        target_dir = f"{out_dir_path}{mapped_label}"
        os.makedirs(target_dir, exist_ok=True)
        
        # 源文件路径
        source_dir = f"{source_base}{subject_id}"
        
        # 检查源文件是否存在
        onset_file = f"{clip_name}_onset.jpg"
        apex_file = f"{clip_name}_apex.jpg"
        offset_file = f"{clip_name}_offset.jpg"
        
        onset_path = f"{source_dir}/{onset_file}"
        apex_path = f"{source_dir}/{apex_file}"
        offset_path = f"{source_dir}/{offset_file}"
        
        if os.path.exists(onset_path) and os.path.exists(apex_path) and os.path.exists(offset_path):
            # 复制三个文件
            shutil.copy(onset_path, f"{target_dir}/{onset_file}")
            shutil.copy(apex_path, f"{target_dir}/{apex_file}")
            shutil.copy(offset_path, f"{target_dir}/{offset_file}")
            
            print(f"Processed: Subject {row['subject']}, File {clip_name}, Original Label {original_label} -> Mapped Label {mapped_label}")
        else:
            print(f"Warning: Missing frames for Subject {row['subject']}, File {clip_name}")
    
    print("SMIC分类完成！")
    print("标签映射:")
    print("  SMIC 0 (negative) -> CASME II 1 (negative)")
    print("  SMIC 1 (positive) -> CASME II 0 (positive)")
    print("  SMIC 2 (surprise) -> CASME II 2 (surprise)")

if __name__ == "__main__":
    classify_frames()
