import os
import shutil
import cv2
import numpy as np
import pandas as pd

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return (x, y)

def computeStrain(u, v):
    u_x= u - pd.DataFrame(u).shift(-1, axis=1)
    v_y= v - pd.DataFrame(v).shift(-1, axis=0)
    u_y= u - pd.DataFrame(u).shift(-1, axis=0)
    v_x= v - pd.DataFrame(v).shift(-1, axis=1)
    os = np.array(np.sqrt(u_x**2 + v_y**2 + 1/2 * (u_y+v_x)**2).ffill(axis=1).ffill(axis=0))
    return os

def calculate_optical_flow(img1, img2):
    frame1 = cv2.imread(img1, 0)
    frame2 = cv2.imread(img2, 0)

    optical_flow = cv2.optflow.DualTVL1OpticalFlow_create()
    flow = optical_flow.calc(frame1, frame2, None)
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    u, v = pol2cart(magnitude, angle)
    os_ = computeStrain(u, v)

    final_u = cv2.resize(u, (48, 48))
    final_v = cv2.resize(v, (48, 48))
    final_os = cv2.resize(os_, (48, 48))

    if ((np.max(final_u) - np.min(final_u))==0):
        normalized_u = final_u.astype(np.uint8)
    else:
        normalized_u = ((final_u - np.min(final_u)) / (np.max(final_u) - np.min(final_u)) * 255).astype(np.uint8)

    if ((np.max(final_v) - np.min(final_v))==0):
        normalized_v = final_v.astype(np.uint8)
    else:
        normalized_v = ((final_v - np.min(final_v)) / (np.max(final_v) - np.min(final_v)) * 255).astype(np.uint8)

    if ((np.max(final_os) - np.min(final_os))==0):
        normalized_os = final_os.astype(np.uint8)
    else:
        normalized_os = ((final_os - np.min(final_os)) / (np.max(final_os) - np.min(final_os)) * 255).astype(np.uint8)

    return normalized_u, normalized_v, normalized_os

def create_loso_structure():
    """
    为SAMM数据集创建LOSO结构并添加光流
    """
    # 源文件夹和目标文件夹
    source_folder = '../../Dataset/SAMM_retinaface_classify'
    target_folder = '../../Dataset/SAMM_retinaface_classify_LOSO'
    
    # 如果目标文件夹已存在，先删除
    if os.path.exists(target_folder):
        shutil.rmtree(target_folder)
        print(f"删除已存在的输出文件夹: {target_folder}")
    
    # 第一步：为分类后的数据集生成光流
    print("开始生成光流...")
    add_optical_flow_to_classified(source_folder)
    print("光流生成完成！")
    
    # 第二步：复制整个原文件夹（包含光流）
    print("复制原数据集...")
    shutil.copytree(source_folder, target_folder)
    
    # 获取所有情感类别
    emotion_classes = ['0', '1', '2', '3', '4', '5', '6', '7']
    
    # 获取所有subject（从文件名中提取）
    subjects = set()
    for emotion_class in emotion_classes:
        emotion_path = os.path.join(target_folder, emotion_class)
        if os.path.exists(emotion_path):
            for file in os.listdir(emotion_path):
                if file.endswith('_onset.jpg'):
                    # 从文件名中提取subject ID
                    subject_id = file.split('_')[0]
                    subjects.add(subject_id)
    
    subjects = sorted(list(subjects))
    print(f"找到的subjects: {subjects}")
    
    # 为每个subject创建LOSO结构
    for subject in subjects:
        print(f"处理 {subject}...")
        
        # 创建subject文件夹
        subject_folder = os.path.join(target_folder, subject)
        os.makedirs(subject_folder, exist_ok=True)
        
        # 创建train和test文件夹
        train_folder = os.path.join(subject_folder, 'train')
        test_folder = os.path.join(subject_folder, 'test')
        os.makedirs(train_folder, exist_ok=True)
        os.makedirs(test_folder, exist_ok=True)
        
        # 为每个情感类别创建文件夹
        for emotion_class in emotion_classes:
            train_emotion_folder = os.path.join(train_folder, emotion_class)
            test_emotion_folder = os.path.join(test_folder, emotion_class)
            os.makedirs(train_emotion_folder, exist_ok=True)
            os.makedirs(test_emotion_folder, exist_ok=True)
        
        # 处理每个情感类别
        for emotion_class in emotion_classes:
            emotion_path = os.path.join(target_folder, emotion_class)
            if not os.path.exists(emotion_path):
                continue
            
            # 获取该类别下属于当前subject的文件
            subject_files = [file for file in os.listdir(emotion_path) if file.startswith(subject)]
            other_files = [file for file in os.listdir(emotion_path) if not file.startswith(subject)]
            
            # 将当前subject的文件放入test文件夹
            for file in subject_files:
                src = os.path.join(emotion_path, file)
                dst = os.path.join(test_folder, emotion_class, file)
                shutil.copy(src, dst)
            
            # 将其他subject的文件放入train文件夹
            for file in other_files:
                src = os.path.join(emotion_path, file)
                dst = os.path.join(train_folder, emotion_class, file)
                shutil.copy(src, dst)
        
        print(f"完成 {subject}")
    
    # 删除原来的情感类别文件夹
    for emotion_class in emotion_classes:
        emotion_path = os.path.join(target_folder, emotion_class)
        if os.path.exists(emotion_path):
            shutil.rmtree(emotion_path)
    
    print("LOSO结构创建完成！")

def add_optical_flow_to_classified(classify_path):
    """
    为分类后的数据集生成光流
    """
    # 遍历每个情感类别
    for emotion_class in os.listdir(classify_path):
        emotion_path = os.path.join(classify_path, emotion_class)
        if not os.path.isdir(emotion_path):
            continue
            
        print(f"Processing emotion class {emotion_class}...")
        
        # 遍历该类别下的所有图片
        for img in os.listdir(emotion_path):
            if '_onset.jpg' in img:  # 只处理onset图片
                base_name = img.replace('_onset.jpg', '')  # 例如：006_006_1_2
                
                # 构建onset、apex、offset文件路径
                onset_path = os.path.join(emotion_path, f"{base_name}_onset.jpg")
                apex_path = os.path.join(emotion_path, f"{base_name}_apex.jpg")
                offset_path = os.path.join(emotion_path, f"{base_name}_offset.jpg")
                
                # 检查三个文件是否都存在
                if os.path.exists(onset_path) and os.path.exists(apex_path) and os.path.exists(offset_path):
                    try:
                        # 计算光流
                        flow_1_u, flow_1_v, flow_1_os = calculate_optical_flow(onset_path, apex_path)
                        flow_2_u, flow_2_v, flow_2_os = calculate_optical_flow(apex_path, offset_path)
                        
                        # 保存光流文件
                        output_filename_1_u = f"{base_name}_1_u.jpg"
                        output_filename_1_v = f"{base_name}_1_v.jpg"
                        output_filename_2_u = f"{base_name}_2_u.jpg"
                        output_filename_2_v = f"{base_name}_2_v.jpg"
                        
                        output_path_1_u = os.path.join(emotion_path, output_filename_1_u)
                        output_path_1_v = os.path.join(emotion_path, output_filename_1_v)
                        output_path_2_u = os.path.join(emotion_path, output_filename_2_u)
                        output_path_2_v = os.path.join(emotion_path, output_filename_2_v)
                        
                        cv2.imwrite(output_path_1_u, flow_1_u)
                        cv2.imwrite(output_path_1_v, flow_1_v)
                        cv2.imwrite(output_path_2_u, flow_2_u)
                        cv2.imwrite(output_path_2_v, flow_2_v)
                        
                    except Exception as e:
                        print(f"Error processing {base_name}: {e}")

def add_optical_flow_to_loso(loso_path):
    """
    为LOSO数据集添加光流
    """
    # 遍历LOSO数据集中的每个主体文件夹
    for subject in os.listdir(loso_path):
        if not subject.startswith('0'):  # SAMM的subject ID是数字
            continue

        print(f"Processing {subject}...")

        # 遍历train和test文件夹
        for split in ['train', 'test']:
            split_path = os.path.join(loso_path, subject, split)

            # 遍历每个表情类别
            for emotion in os.listdir(split_path):
                emotion_path = os.path.join(split_path, emotion)

                # 遍历该类别下的所有图片
                for img in os.listdir(emotion_path):
                    if '_onset.jpg' in img:  # 只处理onset图片
                        base_name = img.replace('_onset.jpg', '')  # 例如：006_006_1_2
                        
                        # 构建onset、apex、offset文件路径
                        onset_path = os.path.join(emotion_path, f"{base_name}_onset.jpg")
                        apex_path = os.path.join(emotion_path, f"{base_name}_apex.jpg")
                        offset_path = os.path.join(emotion_path, f"{base_name}_offset.jpg")
                        
                        # 检查三个文件是否都存在
                        if os.path.exists(onset_path) and os.path.exists(apex_path) and os.path.exists(offset_path):
                            try:
                                # 计算光流
                                flow_1_u, flow_1_v, flow_1_os = calculate_optical_flow(onset_path, apex_path)
                                flow_2_u, flow_2_v, flow_2_os = calculate_optical_flow(apex_path, offset_path)
                                
                                # 保存光流文件
                                output_filename_1_u = f"{base_name}_1_u.jpg"
                                output_filename_1_v = f"{base_name}_1_v.jpg"
                                output_filename_2_u = f"{base_name}_2_u.jpg"
                                output_filename_2_v = f"{base_name}_2_v.jpg"
                                
                                output_path_1_u = os.path.join(emotion_path, output_filename_1_u)
                                output_path_1_v = os.path.join(emotion_path, output_filename_1_v)
                                output_path_2_u = os.path.join(emotion_path, output_filename_2_u)
                                output_path_2_v = os.path.join(emotion_path, output_filename_2_v)
                                
                                cv2.imwrite(output_path_1_u, flow_1_u)
                                cv2.imwrite(output_path_1_v, flow_1_v)
                                cv2.imwrite(output_path_2_u, flow_2_u)
                                cv2.imwrite(output_path_2_v, flow_2_v)
                                
                            except Exception as e:
                                print(f"Error processing {base_name}: {e}")

        print(f"Completed {subject}")

if __name__ == "__main__":
    create_loso_structure()
