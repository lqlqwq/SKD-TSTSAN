import os
import shutil


def add_optical_flow():
    # 定义路径
    loso_path = '../Dataset/CASME2_retinaface_classify_LOSO'
    optflow_path = '../Dataset/CASME2_optflow_retinaface'

    # 遍历LOSO数据集中的每个主体文件夹
    for subject in os.listdir(loso_path):
        if not subject.startswith('sub'):
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
                        base_name = img.replace('_onset.jpg', '') #sub01_EP02_01f`
                        sub_num = base_name.split('_')[0] #sub01
                        # base_name = '_'.join(base_name.split('_')[1:]) #EP02_01f`

                        # 构建光流图片的源路径
                        optflow_source = os.path.join(optflow_path, sub_num)

                        # 构建四个光流文件的名称
                        flow_files = [
                            f"{base_name}_1_u.jpg",
                            f"{base_name}_1_v.jpg",
                            f"{base_name}_2_u.jpg",
                            f"{base_name}_2_v.jpg"
                        ]

                        # 复制光流文件
                        for flow_file in flow_files:
                            flow_file_name = flow_file.split("_",1)[1]
                            src = os.path.join(optflow_source, flow_file_name)
                            dst = os.path.join(emotion_path, flow_file)

                            if os.path.exists(src):
                                shutil.copy2(src, dst)
                            else:
                                print(f"Warning: {src} not found")

        print(f"Completed {subject}")


if __name__ == "__main__":
    add_optical_flow()
