import argparse
from distutils.util import strtobool
from train_classify_SKD_TSTSAN_functions_with_sort_and_timer import main_SKD_TSTSAN_with_Aug_with_SKD


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=strtobool, default=True)
    parser.add_argument('--pre_trained', type=strtobool, default=True)
    parser.add_argument('--Aug_COCO_pre_trained', type=strtobool, default=True)
    parser.add_argument('--save_model', type=strtobool, default=True)
    parser.add_argument('--pre_trained_model_path', type=str, default="Pretrained_model/SKD-TSTSAN.pth", help="path to the model weights pre-trained on macro-expression dataset")
    parser.add_argument('--main_path', type=str, default="Dataset/CASME2_retinaface_classify_LOSO", help="path to the dataset directory")
    parser.add_argument('--exp_name', type=str, default="With_Sort_And_Timer", help="name of the folder to save experimental results")

    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=16)  # 可以修改batch_size
    # parser.add_argument('--batch_size', type=int, default=64)  # 可以修改batch_size
    # parser.add_argument('--learning_rate', type=float, default=0.004)

    parser.add_argument('--seed', type=int, default=1337)
    parser.add_argument('--max_iter', type=int, default=20000)
    parser.add_argument('--model', type=str, default="SKD_TSTSAN")
    parser.add_argument('--loss_function', type=str, default="FocalLoss_weighted")
    parser.add_argument('--class_num', type=int, default=5) #3类还是5类

    parser.add_argument('--temperature', default=3, type=int, help='temperature to smooth the logits')
    parser.add_argument('--alpha', default=0.1, type=float, help='weight of kd loss')
    parser.add_argument('--beta', default=1e-6, type=float, help='weight of feature loss')

    parser.add_argument('--Aug_alpha', type=float, default=2)

    config = parser.parse_args()

    print("=== 训练脚本（带排序和时间估算）===")
    print(f"批次大小: {config.batch_size}")
    print("功能：")
    print("1. 自动排序subjects，确保从sub01开始")
    print("2. 每5秒自动打印训练时间估算")
    print("==================================")

    main_SKD_TSTSAN_with_Aug_with_SKD(config)
