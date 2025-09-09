import os
import shutil
import pandas as pd

def classify_images():
    # 读取CASME2数据集的标注文件
    df = pd.read_excel("../Dataset/CASME2-coding-20140508.xlsx", dtype={"Subject": str})

    # 源文件夹（包含所有帧的文件夹）
    source_base = "../Dataset/CASME2_retinaface"
    # 目标文件夹（只包含特殊帧的文件夹）
    target_base = "../Dataset/CASME2_retinaface_classify"

    # 创建目标文件夹（如果不存在）
    if not os.path.exists(target_base):
        os.makedirs(target_base)

    # 表情类别映射 - 使用固定顺序
    emotion_map = {
        'happiness': "0",
        'surprise': "1",
        'disgust': "2",
        'repression': "3",
        'others': "4"
    }

    # 首先创建表情类别文件夹
    for emotion_idx in emotion_map.values():
        emotion_folder = os.path.join(target_base, emotion_idx)
        os.makedirs(emotion_folder, exist_ok=True)

    # 处理每一行数据
    for index, row in df.iterrows():
        print(row["Subject"])
        subject = f'sub{row["Subject"].zfill(2)}'
        emotion = row['Estimated Emotion'].lower()

        # 跳过不在映射中的表情
        if emotion not in emotion_map:
            continue

        emotion_idx = emotion_map[emotion]

        # 构建源文件夹路径
        source_folder = os.path.join(source_base, subject, row['Filename'])

        if not os.path.exists(source_folder):
            print(f"Warning: Source folder not found: {source_folder}")
            continue

        # 获取onset、apex和offset帧号
        onset = row['OnsetFrame']
        apex = row['ApexFrame']
        offset = row['OffsetFrame']

        # 如果帧号无效，跳过
        if pd.isna(onset) or pd.isna(apex) or pd.isna(offset):
            continue

        # 构建目标文件夹路径
        target_folder = os.path.join(target_base, emotion_idx)

        # 直接复制特殊帧
        source_folder = os.path.join(source_base, subject, row['Filename'])

        onset_path = os.path.join(source_folder, f"img{onset}.jpg")
        apex_path = os.path.join(source_folder, f"img{apex}.jpg")
        offset_path = os.path.join(source_folder, f"img{offset}.jpg")

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

if __name__ == "__main__":
    classify_images()


