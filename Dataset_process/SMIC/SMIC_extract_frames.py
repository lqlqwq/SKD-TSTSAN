import pandas as pd
import shutil
import os
import cv2

def get_frame_format(folder_path, subject):
    """
    检测文件夹中帧号的格式（5位、6位还是8位）
    """
    if not os.path.exists(folder_path):
        return 8  # 默认使用8位
    
    files = os.listdir(folder_path)
    if not files:
        return 8  # 默认使用8位
    
    # 按文件名排序，确保获取第一个文件
    files.sort()
    first_file = files[0]
    print(f"  Debug - first_file: {first_file}")
    if first_file.startswith("reg_image"):
        frame_part = first_file.replace("reg_image", "").replace(".bmp", "")
        print(f"  Debug - frame_part: {frame_part}, length: {len(frame_part)}")
        if len(frame_part) == 5:
            return 5
        elif len(frame_part) == 6:
            return 6
        elif len(frame_part) == 8:
            return 8
    
    return 8  # 默认使用8位

def extract_frames():
    """
    从SMIC数据集中提取onset、apex、offset帧并转换为jpg格式
    """
    # 读取SMIC数据集的标注文件
    df = pd.read_csv("../../Dataset/SMIC_all_cropped/smic_apex.csv")
    
    # 源文件夹和目标文件夹
    dir_path = "../../Dataset/SMIC_all_cropped/"
    out_dir_path = "../../Dataset/SMIC_onset_apex_offset/"
    
    # 如果输出文件夹已存在，先删除
    if os.path.exists(out_dir_path):
        shutil.rmtree(out_dir_path)
        print(f"删除已存在的输出文件夹: {out_dir_path}")
    
    # 创建输出目录
    os.makedirs(out_dir_path, exist_ok=True)
    
    lst = []
    
    # 处理每一行数据
    for index, row in df.iterrows():
        # 构建文件夹路径
        subject_id = f"s{row['subject']}"
        clip_name = row['clip']
        
        # 根据情感标签确定子目录
        if row['label'] == 0:
            emotion_dir = "negative"
        elif row['label'] == 1:
            emotion_dir = "positive"
        elif row['label'] == 2:
            emotion_dir = "surprise"
        else:
            print(f"未知的情感标签: {row['label']}")
            continue
            
        folder_path = f"{dir_path}{subject_id}/micro/{emotion_dir}/{clip_name}"
        
        # 检测该文件夹的帧号格式
        frame_format = get_frame_format(folder_path, subject_id)
        
        # 根据检测到的格式构建文件路径
        if frame_format == 5:
            onset_path = f"{folder_path}/reg_image{row['onset_frame']:05d}.bmp"
            apex_path = f"{folder_path}/reg_image{row['apex_frame']:05d}.bmp"
            offset_path = f"{folder_path}/reg_image{row['offset_frame']:05d}.bmp"
        elif frame_format == 6:
            onset_path = f"{folder_path}/reg_image{row['onset_frame']:06d}.bmp"
            apex_path = f"{folder_path}/reg_image{row['apex_frame']:06d}.bmp"
            offset_path = f"{folder_path}/reg_image{row['offset_frame']:06d}.bmp"
        else:
            onset_path = f"{folder_path}/reg_image{row['onset_frame']}.bmp"
            apex_path = f"{folder_path}/reg_image{row['apex_frame']}.bmp"
            offset_path = f"{folder_path}/reg_image{row['offset_frame']}.bmp"
        
        # 检查三个帧是否都存在
        if os.path.exists(onset_path) and os.path.exists(apex_path) and os.path.exists(offset_path):
            # 创建输出路径
            out_path = f"{out_dir_path}{subject_id}"
            os.makedirs(out_path, exist_ok=True)
            
            # 复制并转换三个关键帧为jpg格式
            frame_mapping = {
                'onset_frame': 'onset',
                'apex_frame': 'apex', 
                'offset_frame': 'offset'
            }
            
            for frame_type, suffix in frame_mapping.items():
                if frame_format == 5:
                    old_path = f"{folder_path}/reg_image{row[frame_type]:05d}.bmp"
                elif frame_format == 6:
                    old_path = f"{folder_path}/reg_image{row[frame_type]:06d}.bmp"
                else:
                    old_path = f"{folder_path}/reg_image{row[frame_type]}.bmp"
                new_path = f"{out_path}/{clip_name}_{suffix}.jpg"
                
                # 读取bmp图像并保存为jpg格式
                image = cv2.imread(old_path)
                cv2.imwrite(new_path, image)
                
            print(f"Processed: Subject {row['subject']}, File {clip_name}")
        else:
            # 如果三个帧中缺少任意一个，跳过整个事件
            missing_frames = []
            if not os.path.exists(onset_path):
                missing_frames.append('onset')
            if not os.path.exists(apex_path):
                missing_frames.append('apex')
            if not os.path.exists(offset_path):
                missing_frames.append('offset')
            
            print(f"Skipped: Subject {row['subject']}, File {clip_name} - Missing frames: {', '.join(missing_frames)}")
            print(f"  Debug - folder_path: {folder_path}")
            print(f"  Debug - frame_format: {frame_format}")
            print(f"  Debug - onset_path: {onset_path}")
            print(f"  Debug - apex_path: {apex_path}")
            print(f"  Debug - offset_path: {offset_path}")
            lst.append(index)
    
    print(f"Total missing: {len(lst)}")
    print("SMIC帧提取和格式转换完成！")

if __name__ == "__main__":
    extract_frames()
