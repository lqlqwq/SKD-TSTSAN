import os
import shutil
import pandas as pd

def classify_images():
    """
    将SAMM数据集按情感类别分类
    """
    # 读取SAMM数据集的标注文件
    df = pd.read_excel("../../Dataset/SAMM/SAMM_Micro_FACS_Codes_v2.xlsx", dtype={"Subject": str})

    # 源文件夹（第一步提取的帧文件夹）
    source_base = "../../Dataset/SAMM_onset_apex_offset"
    # 目标文件夹（按情感分类的文件夹）
    target_base = "../../Dataset/SAMM_retinaface_classify"

    # 如果目标文件夹已存在，先删除
    if os.path.exists(target_base):
        shutil.rmtree(target_base)
        print(f"删除已存在的输出文件夹: {target_base}")

    # 创建目标文件夹（如果不存在）
    os.makedirs(target_base, exist_ok=True)

    # SAMM情感类别映射 - 根据数据统计结果
    emotion_map = {
        'Anger': "0",      # 愤怒
        'Happiness': "1",  # 快乐
        'Surprise': "2",   # 惊讶
        'Disgust': "3",    # 厌恶
        'Fear': "4",       # 恐惧
        'Sadness': "5",    # 悲伤
        'Contempt': "6",   # 轻蔑
        'Other': "7"       # 其他
    }

    # 首先创建情感类别文件夹
    for emotion_idx in emotion_map.values():
        emotion_folder = os.path.join(target_base, emotion_idx)
        os.makedirs(emotion_folder, exist_ok=True)

    # 处理每一行数据
    for index, row in df.iterrows():
        print(f"Processing Subject {int(row['Subject']):03d}")
        subject = f'{int(row["Subject"]):03d}'
        emotion = row['Estimated Emotion']

        # 跳过不在映射中的表情
        if emotion not in emotion_map:
            print(f"Warning: Unknown emotion '{emotion}' for Subject {subject}")
            continue

        emotion_idx = emotion_map[emotion]

        # 构建源文件夹路径（从第一步提取的帧中读取）
        source_folder = os.path.join(source_base, subject)

        if not os.path.exists(source_folder):
            print(f"Warning: Source folder not found: {source_folder}")
            continue

        # 构建目标文件夹路径
        target_folder = os.path.join(target_base, emotion_idx)

        # 复制特殊帧（从第一步提取的帧中复制）
        onset_path = os.path.join(source_folder, f"{row['Filename']}_onset.jpg")
        apex_path = os.path.join(source_folder, f"{row['Filename']}_apex.jpg")
        offset_path = os.path.join(source_folder, f"{row['Filename']}_offset.jpg")

        if os.path.exists(onset_path) and os.path.exists(apex_path) and os.path.exists(offset_path):
            # 复制onset帧
            target_onset_path = os.path.join(target_folder, f"{subject}_{row['Filename']}_onset.jpg")
            shutil.copy2(onset_path, target_onset_path)

            # 复制apex帧
            target_apex_path = os.path.join(target_folder, f"{subject}_{row['Filename']}_apex.jpg")
            shutil.copy2(apex_path, target_apex_path)

            # 复制offset帧
            target_offset_path = os.path.join(target_folder, f"{subject}_{row['Filename']}_offset.jpg")
            shutil.copy2(offset_path, target_offset_path)

            print(f"Processed: Subject {subject}, Emotion {emotion} ({emotion_idx})")
        else:
            print(f"Warning: Missing frames for Subject {subject}, File {row['Filename']}")

if __name__ == "__main__":
    classify_images()
