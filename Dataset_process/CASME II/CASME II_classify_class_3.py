import os
import shutil

def main():
    """
    将5类CASME II数据集转换为3类
    映射规则：
    0 (happiness) -> 0 (positive)
    1 (surprise) -> 2 (surprise)  
    2 (disgust) -> 1 (negative)
    3 (repression) -> 1 (negative)
    4 (others) -> 删除，不参与分类
    """
    
    # 源文件夹和目标文件夹
    source_folder = '../../Dataset/CASME2_retinaface_classify_LOSO'
    target_folder = '../../Dataset/CASME2_retinaface_classify_LOSO_class_3'
    
    # 如果目标文件夹已存在，先删除
    if os.path.exists(target_folder):
        shutil.rmtree(target_folder)
    
    # 复制整个原文件夹
    print("复制原数据集...")
    shutil.copytree(source_folder, target_folder)
    
    # 处理每个subject文件夹
    for sub_num in range(1, 27):
        sub_prefix = f'sub{sub_num:02d}'
        sub_folder = os.path.join(target_folder, sub_prefix)
        
        if not os.path.exists(sub_folder):
            continue
            
        print(f"处理 {sub_prefix}...")
        
        # 处理train文件夹
        train_folder = os.path.join(sub_folder, 'train')
        if os.path.exists(train_folder):
            # 先删除others文件夹（类别4）
            others_path = os.path.join(train_folder, '4')
            if os.path.exists(others_path):
                shutil.rmtree(others_path)
                print(f"  删除 {sub_prefix}/train/4 (others)")
            
            # 使用临时文件夹避免移动冲突
            temp_folder = os.path.join(train_folder, 'temp')
            os.makedirs(temp_folder, exist_ok=True)
            
            # 第一步：将类别1移动到临时文件夹
            class1_path = os.path.join(train_folder, '1')
            if os.path.exists(class1_path):
                temp_class1_path = os.path.join(temp_folder, '1')
                shutil.move(class1_path, temp_class1_path)
            
            # 第二步：将类别2移动到类别1
            class2_path = os.path.join(train_folder, '2')
            if os.path.exists(class2_path):
                new_class1_path = os.path.join(train_folder, '1')
                shutil.move(class2_path, new_class1_path)
            
            # 第三步：将类别3移动到类别1（合并）
            class3_path = os.path.join(train_folder, '3')
            if os.path.exists(class3_path):
                new_class1_path = os.path.join(train_folder, '1')
                # 确保目标文件夹存在
                if not os.path.exists(new_class1_path):
                    os.makedirs(new_class1_path)
                # 移动所有文件到类别1
                for file in os.listdir(class3_path):
                    old_file = os.path.join(class3_path, file)
                    new_file = os.path.join(new_class1_path, file)
                    shutil.move(old_file, new_file)
                os.rmdir(class3_path)
            
            # 第四步：将临时文件夹中的类别1移动到类别2
            temp_class1_path = os.path.join(temp_folder, '1')
            if os.path.exists(temp_class1_path):
                new_class2_path = os.path.join(train_folder, '2')
                shutil.move(temp_class1_path, new_class2_path)
            
            # 删除临时文件夹
            os.rmdir(temp_folder)
        
        # 处理test文件夹
        test_folder = os.path.join(sub_folder, 'test')
        if os.path.exists(test_folder):
            # 先删除others文件夹（类别4）
            others_path = os.path.join(test_folder, '4')
            if os.path.exists(others_path):
                shutil.rmtree(others_path)
                print(f"  删除 {sub_prefix}/test/4 (others)")
            
            # 使用临时文件夹避免移动冲突
            temp_folder = os.path.join(test_folder, 'temp')
            os.makedirs(temp_folder, exist_ok=True)
            
            # 第一步：将类别1移动到临时文件夹
            class1_path = os.path.join(test_folder, '1')
            if os.path.exists(class1_path):
                temp_class1_path = os.path.join(temp_folder, '1')
                shutil.move(class1_path, temp_class1_path)
            
            # 第二步：将类别2移动到类别1
            class2_path = os.path.join(test_folder, '2')
            if os.path.exists(class2_path):
                new_class1_path = os.path.join(test_folder, '1')
                shutil.move(class2_path, new_class1_path)
            
            # 第三步：将类别3移动到类别1（合并）
            class3_path = os.path.join(test_folder, '3')
            if os.path.exists(class3_path):
                new_class1_path = os.path.join(test_folder, '1')
                # 确保目标文件夹存在
                if not os.path.exists(new_class1_path):
                    os.makedirs(new_class1_path)
                # 移动所有文件到类别1
                for file in os.listdir(class3_path):
                    old_file = os.path.join(class3_path, file)
                    new_file = os.path.join(new_class1_path, file)
                    shutil.move(old_file, new_file)
                os.rmdir(class3_path)
            
            # 第四步：将临时文件夹中的类别1移动到类别2
            temp_class1_path = os.path.join(temp_folder, '1')
            if os.path.exists(temp_class1_path):
                new_class2_path = os.path.join(test_folder, '2')
                shutil.move(temp_class1_path, new_class2_path)
            
            # 删除临时文件夹
            os.rmdir(temp_folder)
    
    print("转换完成！")
    print("3类数据集已保存到: Dataset/CASME2_retinaface_classify_LOSO_class3")
    print("使用方法：")
    print("1. 修改训练脚本中的 main_path 为 'Dataset/CASME2_retinaface_classify_LOSO_class3'")
    print("2. 修改 class_num 参数为 3")
    print("3. 运行训练脚本")

if __name__ == "__main__":
    main()
