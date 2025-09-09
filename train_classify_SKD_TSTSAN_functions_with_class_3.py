import os
import sys
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import cv2
import shutil
from collections import OrderedDict
import torch.backends.cudnn as cudnn
import threading

from all_model import get_model, gen_state_dict

CASME2_numbers = [32, 32, 32, 32, 32]

def reset_weights(m):  # Reset the weights for network to avoid weight leakage
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.reset_parameters()

def confusionMatrix(gt, pred, show=False):
    # 计算混淆矩阵
    num_classes = max(max(gt), max(pred)) + 1
    conf_matrix = np.zeros((num_classes, num_classes))
    for i in range(len(gt)):
        conf_matrix[gt[i]][pred[i]] += 1
    
    # 计算F1-score和平均召回率
    f1_scores = []
    recalls = []
    for i in range(num_classes):
        tp = conf_matrix[i][i]
        fp = np.sum(conf_matrix[:, i]) - tp
        fn = np.sum(conf_matrix[i, :]) - tp
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        f1_scores.append(f1)
        recalls.append(recall)
    
    avg_f1 = np.mean(f1_scores)
    avg_recall = np.mean(recalls)
    
    return avg_f1, avg_recall

def normalize_gray(images):
    return images / 255.0

# def recognition_evaluation(dataset, final_gt, final_pred, show=False):
#     # 计算UF1和UAR
#     uf1, uar = confusionMatrix(final_gt, final_pred, show)
    
#     # 计算准确率
#     accuracy = np.sum(np.array(final_gt) == np.array(final_pred)) / len(final_gt)
    
#     if show:
#         print(f"Dataset: {dataset}")
#         print(f"UF1: {uf1:.4f}")
#         print(f"UAR: {uar:.4f}")
#         print(f"Accuracy: {accuracy:.4f}")
    
#     return uf1, uar, accuracy

def recognition_evaluation(dataset, final_gt, final_pred, show=False):
    if dataset == "CASME2":
        label_dict = {'happiness': 0, 'surprise': 1, 'disgust': 2, 'repression': 3, 'others': 4}
    elif dataset == "multi":
        label_dict = {'postive': 0, 'negative': 1, 'surprise': 2}

    f1_list = []
    ar_list = []
    try:
        for emotion, emotion_index in label_dict.items():
            gt_recog = [1 if x == emotion_index else 0 for x in final_gt]
            pred_recog = [1 if x == emotion_index else 0 for x in final_pred]
            try:
                f1_recog, ar_recog = confusionMatrix(gt_recog, pred_recog)
                f1_list.append(f1_recog)
                ar_list.append(ar_recog)
            except Exception as e:
                pass
        UF1 = np.mean(f1_list)
        UAR = np.mean(ar_list)
        
        # 计算准确率
        accuracy = np.sum(np.array(final_gt) == np.array(final_pred)) / len(final_gt)

        if show:
            print(f"Dataset: {dataset}")
            print(f"UF1: {UF1:.4f}")
            print(f"UAR: {UAR:.4f}")
            print(f"Accuracy: {accuracy:.4f}")
        
        
        return UF1, UAR, accuracy
    except:
        return '', '', ''

def extract_prefix(file_name):
    prefixes = ["_1_u", "_2_u", "_1_v", "_2_v", "_apex"]
    for prefix in prefixes:
        if prefix in file_name:
            return file_name.split(prefix)[0]
    return None

def get_folder_all_cases(folder_path):
    unique_prefixes = set()
    for file_name in os.listdir(folder_path):
        if file_name.lower().endswith(".jpg"):
            prefix = extract_prefix(file_name)
            if prefix is not None:
                unique_prefixes.add(prefix)
    unique_prefixes = list(unique_prefixes)
    unique_prefixes.sort()

    return unique_prefixes

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, weight=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, input, target):
        ce_loss = F.cross_entropy(input, target, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

def get_loss_function(loss_name, weight=None):
    if loss_name == "FocalLoss_weighted":
        return FocalLoss(weight=weight)
    elif loss_name == "CrossEntropyLoss":
        return nn.CrossEntropyLoss(weight=weight)
    else:
        return nn.CrossEntropyLoss()

class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

def new_kd_loss_function(output, target_output, temperature):
    # 计算KL散度损失
    log_softmax_output = F.log_softmax(output / temperature, dim=1)
    softmax_target = F.softmax(target_output / temperature, dim=1)
    kl_loss = F.kl_div(log_softmax_output, softmax_target, reduction='batchmean')
    return kl_loss

def feature_loss_function(fea, target_fea):
    # 计算L2损失
    return F.mse_loss(fea, target_fea)

class TrainingTimer:
    """训练时间估算器"""
    def __init__(self, total_epochs, current_subject, total_subjects, data_loading_time=0):
        self.start_time = time.time()
        self.data_loading_time = data_loading_time  # 数据加载时间
        self.training_start_time = None      # 训练开始时间
        self.total_epochs = total_epochs
        self.current_subject = current_subject
        self.total_subjects = total_subjects
        self.current_epoch = 0
        self.running = True
        
        # 启动定时器线程
        self.timer_thread = threading.Thread(target=self._timer_loop)
        self.timer_thread.daemon = True
        self.timer_thread.start()
    
    def start_training(self):
        """开始训练时调用"""
        self.training_start_time = time.time()
    
    def update_epoch(self, epoch):
        self.current_epoch = epoch
    
    def _timer_loop(self):
        """每5分钟打印一次时间估算"""
        while self.running:
            time.sleep(5 * 60)
            if self.running:
                self._print_estimate()
    
    def _print_estimate(self):
        """打印时间估算"""
        total_elapsed_time = time.time() - self.start_time
        
        # 格式化时间
        def format_time(seconds):
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = int(seconds % 60)
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        
        print(f"\n=== 时间估算 (Subject {self.current_subject}/{self.total_subjects}) ===")
        print(f"当前Epoch: {self.current_epoch}/{self.total_epochs}")
        print(f"总已用时间: {format_time(total_elapsed_time)}")
        
        # 显示数据加载时间
        print(f"数据加载时间: {format_time(self.data_loading_time)}")
        
        # 显示训练时间
        if self.current_epoch > 0 and self.training_start_time is not None:
            # 只计算训练时间，不包括数据加载时间
            training_elapsed_time = time.time() - self.training_start_time
            
            # 计算每个epoch的平均训练时间
            time_per_epoch = training_elapsed_time / self.current_epoch
            
            # 计算当前subject剩余epochs
            remaining_epochs_current = self.total_epochs - self.current_epoch
            
            # 计算当前subject剩余时间
            remaining_time_current = remaining_epochs_current * time_per_epoch
            
            # 计算剩余subjects的总时间
            remaining_subjects = self.total_subjects - self.current_subject
            total_remaining_time = remaining_time_current + (remaining_subjects * self.total_epochs * time_per_epoch)
            
            print(f"训练已用时间: {format_time(training_elapsed_time)}")
            print(f"每Epoch平均时间: {time_per_epoch:.1f}秒")
            print(f"当前Subject剩余时间: {format_time(remaining_time_current)}")
            print(f"预计完成时间: {format_time(total_remaining_time)}")
        
        print("=" * 50)
    
    def stop(self):
        """停止时间估算器"""
        self.running = False

def main_SKD_TSTSAN_with_Aug_with_SKD(config):
    # 设置随机种子
    seed = config.seed
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True

    is_cuda = torch.cuda.is_available()
    if is_cuda:
        device = torch.device('cuda:0')  # 限制为GPU 0
        torch.cuda.set_device(0)  # 设置默认GPU为0
    else:
        # device = torch.device('cpu')
        raise Exception("No GPU")

    if (config.train):
        if not os.path.exists('./Experiment_for_recognize/' + config.exp_name):
            os.makedirs('./Experiment_for_recognize/' + config.exp_name)

    current_file = os.path.abspath(__file__)
    shutil.copy(current_file, './Experiment_for_recognize/' + config.exp_name)
    shutil.copy("./all_model.py", './Experiment_for_recognize/' + config.exp_name)

    log_file_path = './Experiment_for_recognize/' + config.exp_name + "/log.txt"
    sys.stdout = Logger(log_file_path)

    total_gt = []
    total_pred = []
    best_total_pred = []
    all_accuracy_dict = {}

    t = time.time()

    main_path = config.main_path
    subName = os.listdir(main_path)
    
    # 1. 对subjects进行排序，确保从sub01开始
    subName.sort()
    print(f"Subjects will be trained in order: {subName}")
    total_subjects = len(subName)

    for sub_idx, n_subName in enumerate(subName, 1):
        print(f'\nSubject: {n_subName} ({sub_idx}/{total_subjects})')

        # 开始数据加载计时
        data_loading_start = time.time()

        X_train = []
        y_train = []

        X_test = []
        y_test = []

        expression = os.listdir(main_path + '/' + n_subName + '/train')
        for n_expression in expression:
            case_list = get_folder_all_cases(main_path + '/' + n_subName + '/train/' + n_expression)

            for case in case_list:
                y_train.append(int(n_expression))

                end_input = []
                large_S = normalize_gray(
                    cv2.imread(main_path + '/' + n_subName + '/train/' + n_expression + '/' + case + "_apex.jpg", 0))
                large_S_onset = normalize_gray(
                    cv2.imread(main_path + '/' + n_subName + '/train/' + n_expression + '/' + case + "_onset.jpg", 0))
                small_S = cv2.resize(large_S, (48, 48))
                small_S_onset = cv2.resize(large_S_onset, (48, 48))
                end_input.append(small_S)
                end_input.append(small_S_onset)

                grid_sizes = [4]
                for grid_size in grid_sizes:
                    height, width = large_S.shape
                    block_height, block_width = height // grid_size, width // grid_size

                    for i in range(grid_size):
                        for j in range(grid_size):
                            block = large_S[i * block_height: (i + 1) * block_height,
                                    j * block_width: (j + 1) * block_width]

                            scaled_block = cv2.resize(block, (48, 48))

                            end_input.append(scaled_block)

                for grid_size in grid_sizes:
                    height, width = large_S.shape
                    block_height, block_width = height // grid_size, width // grid_size

                    for i in range(grid_size):
                        for j in range(grid_size):
                            block = large_S_onset[i * block_height: (i + 1) * block_height,
                                    j * block_width: (j + 1) * block_width]

                            scaled_block = cv2.resize(block, (48, 48))

                            end_input.append(scaled_block)

                end_input.append(normalize_gray(
                    cv2.imread(main_path + '/' + n_subName + '/train/' + n_expression + '/' + case + "_1_u.jpg", 0)))
                end_input.append(normalize_gray(
                    cv2.imread(main_path + '/' + n_subName + '/train/' + n_expression + '/' + case + "_1_v.jpg", 0)))
                end_input.append(normalize_gray(
                    cv2.imread(main_path + '/' + n_subName + '/train/' + n_expression + '/' + case + "_2_u.jpg", 0)))
                end_input.append(normalize_gray(
                    cv2.imread(main_path + '/' + n_subName + '/train/' + n_expression + '/' + case + "_2_v.jpg", 0)))

                X_train.append(np.array(end_input))

        expression = os.listdir(main_path + '/' + n_subName + '/test')
        for n_expression in expression:
            case_list = get_folder_all_cases(main_path + '/' + n_subName + '/test/' + n_expression)

            for case in case_list:
                y_test.append(int(n_expression))

                end_input = []
                large_S = normalize_gray(
                    cv2.imread(main_path + '/' + n_subName + '/test/' + n_expression + '/' + case + "_apex.jpg", 0))
                large_S_onset = normalize_gray(
                    cv2.imread(main_path + '/' + n_subName + '/test/' + n_expression + '/' + case + "_onset.jpg", 0))
                small_S = cv2.resize(large_S, (48, 48))
                small_S_onset = cv2.resize(large_S_onset, (48, 48))
                end_input.append(small_S)
                end_input.append(small_S_onset)

                grid_sizes = [4]
                for grid_size in grid_sizes:
                    height, width = large_S.shape
                    block_height, block_width = height // grid_size, width // grid_size

                    for i in range(grid_size):
                        for j in range(grid_size):
                            block = large_S[i * block_height: (i + 1) * block_height,
                                    j * block_width: (j + 1) * block_width]

                            scaled_block = cv2.resize(block, (48, 48))

                            end_input.append(scaled_block)

                for grid_size in grid_sizes:
                    height, width = large_S.shape
                    block_height, block_width = height // grid_size, width // grid_size

                    for i in range(grid_size):
                        for j in range(grid_size):
                            block = large_S_onset[i * block_height: (i + 1) * block_height,
                                    j * block_width: (j + 1) * block_width]

                            scaled_block = cv2.resize(block, (48, 48))

                            end_input.append(scaled_block)

                end_input.append(normalize_gray(
                    cv2.imread(main_path + '/' + n_subName + '/test/' + n_expression + '/' + case + "_1_u.jpg", 0)))
                end_input.append(normalize_gray(
                    cv2.imread(main_path + '/' + n_subName + '/test/' + n_expression + '/' + case + "_1_v.jpg", 0)))
                end_input.append(normalize_gray(
                    cv2.imread(main_path + '/' + n_subName + '/test/' + n_expression + '/' + case + "_2_u.jpg", 0)))
                end_input.append(normalize_gray(
                    cv2.imread(main_path + '/' + n_subName + '/test/' + n_expression + '/' + case + "_2_v.jpg", 0)))

                X_test.append(np.array(end_input))

        X_train = np.array(X_train)
        y_train = np.array(y_train)
        X_test = np.array(X_test)
        y_test = np.array(y_test)

        train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
        test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))

        train_dl = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
        test_dl = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

        # 结束数据加载计时
        data_loading_end = time.time()
        data_loading_time = data_loading_end - data_loading_start
        print(f"数据加载完成，耗时: {data_loading_time:.2f}秒")

        weight_path = './Experiment_for_recognize/' + config.exp_name + '/' + n_subName + '/' + n_subName + '.pth'

        model = get_model(config.model, config.class_num, config.Aug_alpha).to(device)

        if (config.train):
            if (config.pre_trained):
                model.apply(reset_weights)
                pre_trained_model = torch.load(config.pre_trained_model_path)
                filtered_dict = OrderedDict((k, v) for k, v in pre_trained_model.items() if (not "fc" in k))
                model.load_state_dict(filtered_dict, strict=False)
            elif (config.Aug_COCO_pre_trained):
                model.apply(reset_weights)
                Aug_weight_path = r"motion_magnification_learning_based_master/magnet.pth"
                Aug_state_dict = gen_state_dict(Aug_weight_path)
                model.Aug_Encoder_L.load_state_dict(Aug_state_dict, strict=False)
                model.Aug_Encoder_S.load_state_dict(Aug_state_dict, strict=False)
                model.Aug_Encoder_T.load_state_dict(Aug_state_dict, strict=False)
                model.Aug_Manipulator_L.load_state_dict(Aug_state_dict, strict=False)
                model.Aug_Manipulator_S.load_state_dict(Aug_state_dict, strict=False)
                model.Aug_Manipulator_T.load_state_dict(Aug_state_dict, strict=False)
            else:
                model.apply(reset_weights)

        else:
            model.load_state_dict(torch.load(weight_path))

        # 在模型加载后重新创建损失函数，确保权重张量维度正确
        if config.loss_function == "FocalLoss_weighted":
            # 检查是否是CASME2的3分类版本
            if config.main_path.split("/")[1].split("_")[0] == "CASME2" and "class_3" in config.main_path:
                # CASME2 3分类版本
                numbers = [1, 1, 1]  # 3分类的权重
            elif config.main_path.split("/")[1].split("_")[0] == "CASME2":
                # CASME2 5分类版本
                numbers = CASME2_numbers
            else:
                # 对于其他数据集，根据class_num动态设置权重
                if config.class_num == 3:
                    numbers = [1, 1, 1]  # 3分类的默认权重
                elif config.class_num == 5:
                    numbers = [1, 1, 1, 1, 1]  # 5分类的默认权重
                else:
                    numbers = [1] * config.class_num  # 通用情况

            sum_reciprocal = sum(1 / num for num in numbers)
            weights = [(1 / num) / sum_reciprocal for num in numbers]

            loss_fn = get_loss_function(config.loss_function, torch.tensor(weights).to(device))
        else:
            loss_fn = get_loss_function(config.loss_function)

        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, betas=(0.9, 0.99), weight_decay=0.0005)

        best_accuracy_for_each_subject = 0
        best_each_subject_pred = []

        max_iter = config.max_iter
        iter_num = 0
        epochs = max_iter // len(train_dl) + 1

        # 创建训练时间估算器，传入数据加载时间
        timer = TrainingTimer(epochs, sub_idx, total_subjects, data_loading_time)
        print(f"Starting training for {epochs} epochs...")

        for epoch in range(1, epochs + 1):
            # 在第一个epoch开始时启动训练计时
            if epoch == 1:
                timer.start_training()
            # 更新当前epoch
            timer.update_epoch(epoch)
            
            if (config.train):
                model.train()
                train_ce_loss = 0.0
                middle_loss1 = 0.0
                middle_loss2 = 0.0
                KL_loss1 = 0.0
                KL_loss2 = 0.0
                L2_loss1 = 0.0
                L2_loss2 = 0.0
                loss_sum = 0.0

                num_train_correct = 0
                num_train_examples = 0

                middle1_num_train_correct = 0
                middle2_num_train_correct = 0

                for batch_idx, batch in enumerate(train_dl):
                    optimizer.zero_grad()
                    x = batch[0].to(device)
                    y = batch[1].to(device)
                    yhat, AC1_out, AC2_out, final_feature, AC1_feature, AC2_feature = model(x)
                    loss = loss_fn(yhat, y)
                    AC1_loss = loss_fn(AC1_out, y)
                    AC2_loss = loss_fn(AC2_out, y)
                    temperature = config.temperature
                    temp4 = yhat / temperature
                    temp4 = torch.softmax(temp4, dim=1)
                    loss1by4 = new_kd_loss_function(AC1_out, temp4.detach(), temperature) * (temperature ** 2)
                    loss2by4 = new_kd_loss_function(AC2_out, temp4.detach(), temperature) * (temperature ** 2)
                    feature_loss_1 = feature_loss_function(AC1_feature, final_feature.detach())
                    feature_loss_2 = feature_loss_function(AC2_feature, final_feature.detach())

                    total_losses = loss + (1 - config.alpha) * (AC1_loss + AC2_loss) + \
                                   config.alpha * (loss1by4 + loss2by4) + \
                                   config.beta * (feature_loss_1 + feature_loss_2)

                    total_losses.backward()
                    optimizer.step()

                    train_ce_loss += loss.data.item() * x.size(0)
                    middle_loss1 += AC1_loss.data.item() * x.size(0)
                    middle_loss2 += AC2_loss.data.item() * x.size(0)
                    KL_loss1 += loss1by4.data.item() * x.size(0)
                    KL_loss2 += loss2by4.data.item() * x.size(0)
                    L2_loss1 += feature_loss_1.data.item() * x.size(0)
                    L2_loss2 += feature_loss_2.data.item() * x.size(0)
                    loss_sum += total_losses * x.size(0)

                    num_train_correct += (torch.max(yhat, 1)[1] == y).sum().item()
                    num_train_examples += x.shape[0]

                    middle1_num_train_correct += (torch.max(AC1_out, 1)[1] == y).sum().item()
                    middle2_num_train_correct += (torch.max(AC2_out, 1)[1] == y).sum().item()

                    iter_num += 1
                    if iter_num >= max_iter:
                        break

                train_acc = num_train_correct / num_train_examples
                middle1_acc = middle1_num_train_correct / num_train_examples
                middle2_acc = middle2_num_train_correct / num_train_examples

                train_ce_loss = train_ce_loss / len(train_dl.dataset)
                middle_loss1 = middle_loss1 / len(train_dl.dataset)
                middle_loss2 = middle_loss2 / len(train_dl.dataset)
                KL_loss1 = KL_loss1 / len(train_dl.dataset)
                KL_loss2 = KL_loss2 / len(train_dl.dataset)
                L2_loss1 = L2_loss1 / len(train_dl.dataset)
                L2_loss2 = L2_loss2 / len(train_dl.dataset)
                loss_sum = loss_sum / len(train_dl.dataset)

            model.eval()
            num_val_correct = 0

            middle1_num_val_correct = 0
            middle2_num_val_correct = 0

            num_val_examples = 0

            with torch.no_grad():
                for batch in test_dl:
                    x = batch[0].to(device)
                    y = batch[1].to(device)
                    yhat, AC1_out, AC2_out, final_feature, AC1_feature, AC2_feature = model(x)

                    num_val_correct += (torch.max(yhat, 1)[1] == y).sum().item()
                    num_val_examples += x.shape[0]

                    middle1_num_val_correct += (torch.max(AC1_out, 1)[1] == y).sum().item()
                    middle2_num_val_correct += (torch.max(AC2_out, 1)[1] == y).sum().item()

            val_acc = num_val_correct / num_val_examples if num_val_examples > 0 else 0.0
            middle1_val_acc = middle1_num_val_correct / num_val_examples if num_val_examples > 0 else 0.0
            middle2_val_acc = middle2_num_val_correct / num_val_examples if num_val_examples > 0 else 0.0

            if num_val_examples == 0:
                print("警告: 测试集为空，跳过此subject")
                continue

            print(f"Epoch {epoch}: Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

            if val_acc > best_accuracy_for_each_subject:
                best_accuracy_for_each_subject = val_acc
                best_each_subject_pred = []
                with torch.no_grad():
                    for batch in test_dl:
                        x = batch[0].to(device)
                        y = batch[1].to(device)
                        yhat, AC1_out, AC2_out, final_feature, AC1_feature, AC2_feature = model(x)
                        pred = torch.max(yhat, 1)[1]
                        best_each_subject_pred.extend(pred.cpu().numpy())

            if iter_num >= max_iter:
                break

        # 停止时间估算器
        timer.stop()

        # 保存模型
        if config.save_model:
            # 创建保存目录
            save_dir = os.path.dirname(weight_path)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            torch.save(model.state_dict(), weight_path)

        # 评估最佳结果
        best_each_subject_gt = y_test
        # UF1, UAR, accuracy = recognition_evaluation("CASME2", best_each_subject_gt, best_each_subject_pred, show=True)
        UF1, UAR, accuracy = recognition_evaluation("multi", best_each_subject_gt, best_each_subject_pred, show=True)

        total_gt.extend(best_each_subject_gt)
        total_pred.extend(best_each_subject_pred)

        all_accuracy_dict[n_subName] = best_accuracy_for_each_subject

        # print(f"Subject {n_subName} completed. Best accuracy: {best_accuracy_for_each_subject:.4f}")

    # 计算总体结果
    # UF1, UAR, accuracy = recognition_evaluation("CASME2", total_gt, total_pred, show=True)
    UF1, UAR, accuracy = recognition_evaluation("multi", total_gt, total_pred, show=True)

    print(f"Overall Results:")
    print(f"UF1: {UF1:.4f}")
    print(f"UAR: {UAR:.4f}")
    print(f"Accuracy: {accuracy:.4f}")

    # 计算平均准确率
    avg_accuracy = np.mean(list(all_accuracy_dict.values()))
    print(f"Average Accuracy: {avg_accuracy:.4f}")

    print(f"Total training time: {time.time() - t:.2f} seconds")
