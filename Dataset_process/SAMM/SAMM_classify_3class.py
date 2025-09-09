import os
import shutil

def main():
    """
    将8类SAMM数据集转换为3类
    映射规则（按照CASME II格式）：
    0 (Anger) -> 1 (negative)
    1 (Happiness) -> 0 (positive)
    2 (Surprise) -> 2 (surprise)
    3 (Disgust) -> 1 (negative)
    4 (Fear) -> 1 (negative)
    5 (Sadness) -> 1 (negative)
    6 (Contempt) -> 1 (negative)
    7 (Other) -> 删除，不参与分类
    """
    
    # 源文件夹和目标文件夹
    source_folder = '../../Dataset/SAMM_retinaface_classify_LOSO'
    target_folder = '../../Dataset/SAMM_retinaface_classify_LOSO_class3'
    
    # 如果目标文件夹已存在，先删除
    if os.path.exists(target_folder):
        shutil.rmtree(target_folder)
        print(f"删除已存在的输出文件夹: {target_folder}")
    
    # 复制整个原文件夹
    print("复制原数据集...")
    shutil.copytree(source_folder, target_folder)
    
    # 获取所有subjects
    subjects = [d for d in os.listdir(target_folder) if os.path.isdir(os.path.join(target_folder, d))]
    subjects = sorted(subjects)
    
    print(f"找到的subjects: {subjects}")
    
    # 处理每个subject文件夹
    for subject in subjects:
        print(f"处理 {subject}...")
        sub_folder = os.path.join(target_folder, subject)
        
        # 处理train文件夹
        train_folder = os.path.join(sub_folder, 'train')
        if os.path.exists(train_folder):
            # 先删除others文件夹（类别7）
            others_path = os.path.join(train_folder, '7')
            if os.path.exists(others_path):
                shutil.rmtree(others_path)
                print(f"  删除 {subject}/train/7 (others)")
            
            # 使用临时文件夹避免移动冲突
            temp_folder = os.path.join(train_folder, 'temp')
            os.makedirs(temp_folder, exist_ok=True)
            
            # 第一步：将类别1移动到临时文件夹（Happiness -> positive）
            class1_path = os.path.join(train_folder, '1')
            if os.path.exists(class1_path):
                temp_class1_path = os.path.join(temp_folder, '1')
                shutil.move(class1_path, temp_class1_path)
            
            # 第二步：将类别2移动到临时文件夹（Surprise -> surprise）
            class2_path = os.path.join(train_folder, '2')
            if os.path.exists(class2_path):
                temp_class2_path = os.path.join(temp_folder, '2')
                shutil.move(class2_path, temp_class2_path)
            
            # 第三步：将类别0,3,4,5,6移动到类别1（negative）
            negative_classes = ['0', '3', '4', '5', '6']
            new_class1_path = os.path.join(train_folder, '1')
            if not os.path.exists(new_class1_path):
                os.makedirs(new_class1_path)
            
            for old_class in negative_classes:
                old_class_path = os.path.join(train_folder, old_class)
                if os.path.exists(old_class_path):
                    # 移动所有文件到类别1
                    for file in os.listdir(old_class_path):
                        old_file = os.path.join(old_class_path, file)
                        new_file = os.path.join(new_class1_path, file)
                        shutil.move(old_file, new_file)
                    os.rmdir(old_class_path)
            
            # 第四步：将临时文件夹中的类别1移动到类别0（positive）
            temp_class1_path = os.path.join(temp_folder, '1')
            if os.path.exists(temp_class1_path):
                new_class0_path = os.path.join(train_folder, '0')
                shutil.move(temp_class1_path, new_class0_path)
            
            # 第五步：将临时文件夹中的类别2移动到类别2（surprise）
            temp_class2_path = os.path.join(temp_folder, '2')
            if os.path.exists(temp_class2_path):
                new_class2_path = os.path.join(train_folder, '2')
                shutil.move(temp_class2_path, new_class2_path)
            
            # 删除临时文件夹
            os.rmdir(temp_folder)
        
        # 处理test文件夹
        test_folder = os.path.join(sub_folder, 'test')
        if os.path.exists(test_folder):
            # 先删除others文件夹（类别7）
            others_path = os.path.join(test_folder, '7')
            if os.path.exists(others_path):
                shutil.rmtree(others_path)
                print(f"  删除 {subject}/test/7 (others)")
            
            # 使用临时文件夹避免移动冲突
            temp_folder = os.path.join(test_folder, 'temp')
            os.makedirs(temp_folder, exist_ok=True)
            
            # 第一步：将类别1移动到临时文件夹（Happiness -> positive）
            class1_path = os.path.join(test_folder, '1')
            if os.path.exists(class1_path):
                temp_class1_path = os.path.join(temp_folder, '1')
                shutil.move(class1_path, temp_class1_path)
            
            # 第二步：将类别2移动到临时文件夹（Surprise -> surprise）
            class2_path = os.path.join(test_folder, '2')
            if os.path.exists(class2_path):
                temp_class2_path = os.path.join(temp_folder, '2')
                shutil.move(class2_path, temp_class2_path)
            
            # 第三步：将类别0,3,4,5,6移动到类别1（negative）
            negative_classes = ['0', '3', '4', '5', '6']
            new_class1_path = os.path.join(test_folder, '1')
            if not os.path.exists(new_class1_path):
                os.makedirs(new_class1_path)
            
            for old_class in negative_classes:
                old_class_path = os.path.join(test_folder, old_class)
                if os.path.exists(old_class_path):
                    # 移动所有文件到类别1
                    for file in os.listdir(old_class_path):
                        old_file = os.path.join(old_class_path, file)
                        new_file = os.path.join(new_class1_path, file)
                        shutil.move(old_file, new_file)
                    os.rmdir(old_class_path)
            
            # 第四步：将临时文件夹中的类别1移动到类别0（positive）
            temp_class1_path = os.path.join(temp_folder, '1')
            if os.path.exists(temp_class1_path):
                new_class0_path = os.path.join(test_folder, '0')
                shutil.move(temp_class1_path, new_class0_path)
            
            # 第五步：将临时文件夹中的类别2移动到类别2（surprise）
            temp_class2_path = os.path.join(temp_folder, '2')
            if os.path.exists(temp_class2_path):
                new_class2_path = os.path.join(test_folder, '2')
                shutil.move(temp_class2_path, new_class2_path)
            
            # 删除临时文件夹
            os.rmdir(temp_folder)
    
    print("转换完成！")
    print("3类数据集已保存到: Dataset/SAMM_retinaface_classify_LOSO_class3")
    print("使用方法：")
    print("1. 修改训练脚本中的 main_path 为 'Dataset/SAMM_retinaface_classify_LOSO_class3'")
    print("2. 修改 class_num 参数为 3")
    print("3. 运行训练脚本")

if __name__ == "__main__":
    main()
