import pandas as pd
import shutil
import os
import cv2
import sys
sys.path.append('../../')
from RetinaFace.tools import FaceDetector
import faulthandler
faulthandler.enable()

def get_frame_format(folder_path, subject):
    """
    检测文件夹中帧号的格式（4位还是5位）
    """
    if not os.path.exists(folder_path):
        return 5  # 默认使用5位
    
    files = os.listdir(folder_path)
    if not files:
        return 5  # 默认使用5位
    
    # 检查第一个文件的格式
    first_file = files[0]
    if first_file.startswith(f"{subject}_"):
        frame_part = first_file.replace(f"{subject}_", "").replace(".jpg", "")
        if len(frame_part) == 5:
            return 5
        elif len(frame_part) == 4:
            return 4
    
    return 5  # 默认使用5位

def extract_frames():
    """
    从SAMM数据集中提取onset、apex、offset帧并裁剪人脸
    """
    # 初始化人脸检测器
    face_det_model_path = "../../RetinaFace/Resnet50_Final.pth"
    face_detection = FaceDetector(face_det_model_path)
    
    # 读取SAMM数据集的标注文件
    df = pd.read_excel("../../Dataset/SAMM/SAMM_Micro_FACS_Codes_v2.xlsx", dtype={"Subject": str})
    
    # 源文件夹和目标文件夹
    dir_path = "../../Dataset/SAMM/"
    out_dir_path = "../../Dataset/SAMM_onset_apex_offset/"
    
    # 如果输出文件夹已存在，先删除
    if os.path.exists(out_dir_path):
        shutil.rmtree(out_dir_path)
        print(f"删除已存在的输出文件夹: {out_dir_path}")
    
    # 创建输出目录
    os.makedirs(out_dir_path, exist_ok=True)
    
    lst = []
    
    def is_number(x):
        try:
            float(x)
            return True
        except (ValueError, TypeError):
            return False
    
    # 处理每一行数据
    for index, row in df.iterrows():
        # 构建文件夹路径
        folder_path = f"{dir_path}{int(row['Subject']):03d}/{row['Filename']}"
        
        # 检测该文件夹的帧号格式
        subject_id = f"{int(row['Subject']):03d}"
        frame_format = get_frame_format(folder_path, subject_id)
        
        # 根据检测到的格式构建文件路径
        Onset_path = f"{folder_path}/{subject_id}_{int(row['Onset Frame']):0{frame_format}d}.jpg"
        Offset_path = f"{folder_path}/{subject_id}_{int(row['Offset Frame']):0{frame_format}d}.jpg"
        Apex_path = f"{folder_path}/{subject_id}_{int(row['Apex Frame']):0{frame_format}d}.jpg"
        
        # 检查三个帧是否都存在
        if os.path.exists(Onset_path) and os.path.exists(Offset_path) and os.path.exists(Apex_path):
            # 创建输出路径
            out_path = f"{out_dir_path}{int(row['Subject']):03d}"
            os.makedirs(out_path, exist_ok=True)
            
            # 获取人脸边界框（使用第一帧）
            image = cv2.imread(Onset_path)
            face_left, face_top, face_right, face_bottom = face_detection.cal(image)
            
            # 复制并裁剪三个关键帧
            frame_mapping = {
                'Onset Frame': 'onset',
                'Apex Frame': 'apex', 
                'Offset Frame': 'offset'
            }
            
            for frame_type, suffix in frame_mapping.items():
                old_path = f"{folder_path}/{subject_id}_{int(row[frame_type]):0{frame_format}d}.jpg"
                new_path = f"{out_path}/{row['Filename']}_{suffix}.jpg"
                
                # 读取图像并裁剪人脸
                image = cv2.imread(old_path)
                face = image[face_top:face_bottom + 1, face_left:face_right + 1, :]
                face = cv2.resize(face, (128, 128))
                
                # 保存裁剪后的图像
                cv2.imwrite(new_path, face)
                
            print(f"Processed: Subject {row['Subject']}, File {row['Filename']}")
        else:
            # 如果三个帧中缺少任意一个，跳过整个事件
            missing_frames = []
            if not os.path.exists(Onset_path):
                missing_frames.append('onset')
            if not os.path.exists(Apex_path):
                missing_frames.append('apex')
            if not os.path.exists(Offset_path):
                missing_frames.append('offset')
            
            print(f"Skipped: Subject {row['Subject']}, File {row['Filename']} - Missing frames: {', '.join(missing_frames)}")
            lst.append(index)
    
    print(f"Total missing: {len(lst)}")

if __name__ == "__main__":
    extract_frames()
