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

def recognition_evaluation(dataset, final_gt, final_pred, show=False):
    # 计算UF1和UAR
    uf1, uar = confusionMatrix(final_gt, final_pred, show)
    
    # 计算准确率
    accuracy = np.sum(np.array(final_gt) == np.array(final_pred)) / len(final_gt)
    
    if show:
        print(f"Dataset: {dataset}")
        print(f"UF1: {uf1:.4f}")
        print(f"UAR: {uar:.4f}")
        print(f"Accuracy: {accuracy:.4f}")
    
    return uf1, uar, accuracy

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
    def __init__(self, total_epochs, current_subject, total_subjects):
        self.start_time = time.time()
        self.total_epochs = total_epochs
        self.current_subject = current_subject
        self.total_subjects = total_subjects
        self.current_epoch = 0
        self.running = True
        
        # 启动定时器线程
        self.timer_thread = threading.Thread(target=self._timer_loop)
        self.timer_thread.daemon = True
        self.timer_thread.start()
    
    def update_epoch(self, epoch):
        self.current_epoch = epoch
    
    def _timer_loop(self):
        """每5秒打印一次时间估算"""
        while self.running:
            time.sleep(5)
            if self.running:
                self._print_estimate()
    
    def _print_estimate(self):
        """打印时间估算"""
        elapsed_time = time.time() - self.start_time
        
        if self.current_epoch > 0:
            # 计算每个epoch的平均时间
            time_per_epoch = elapsed_time / self.current_epoch
            
            # 计算当前subject剩余epochs
            remaining_epochs_current = self.total_epochs - self.current_epoch
            
            # 计算当前subject剩余时间
            remaining_time_current = remaining_epochs_current * time_per_epoch
            
            # 计算剩余subjects的总时间
            remaining_subjects = self.total_subjects - self.current_subject
            total_remaining_time = remaining_time_current + (remaining_subjects * self.total_epochs * time_per_epoch)
            
            # 格式化时间
            def format_time(seconds):
                hours = int(seconds // 3600)
                minutes = int((seconds % 3600) // 60)
                secs = int(seconds % 60)
                return f"{hours:02d}:{minutes:02d}:{secs:02d}"
            
            print(f"\n=== 时间估算 (Subject {self.current_subject}/{self.total_subjects}) ===")
            print(f"当前Epoch: {self.current_epoch}/{self.total_epochs}")
            print(f"已用时间: {format_time(elapsed_time)}")
            print(f"当前Subject剩余时间: {format_time(remaining_time_current)}")
            print(f"总剩余时间: {format_time(total_remaining_time)}")
            print(f"预计完成时间: {format_time(elapsed_time + total_remaining_time)}")
            print("=" * 50)
    
    def stop(self):
        self.running = False

def main_SKD_TSTSAN_with_Aug_with_SKD(config):
    learning_rate = config.learning_rate
    batch_size = config.batch_size

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
        device = torch.device('cuda')
    else:
        # device = torch.device('cpu')
        raise Exception("No GPU")

    if config.loss_function == "FocalLoss_weighted":
        if config.main_path.split("/")[1].split("_")[0] == "CASME2":
            numbers = CASME2_numbers

        sum_reciprocal = sum(1 / num for num in numbers)
        weights = [(1 / num) / sum_reciprocal for num in numbers]

        loss_fn = get_loss_function(config.loss_function, torch.tensor(weights).to(device))
    else:
        loss_fn = get_loss_function(config.loss_function)

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

                end_input = np.stack(end_input, axis=-1)
                X_train.append(end_input)

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

                end_input = np.stack(end_input, axis=-1)
                X_test.append(end_input)

        # X_train = torch.Tensor(X_train).permute(0, 3, 1, 2)
        X_train = np.array(X_train)
        X_train = torch.from_numpy(X_train).float().permute(0, 3, 1, 2)
        y_train = torch.Tensor(y_train).to(dtype=torch.long)
        dataset_train = TensorDataset(X_train, y_train)

        def worker_init_fn(worker_id):
            random.seed(seed + worker_id)
            np.random.seed(seed + worker_id)

        # train_dl = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True,
        #                       worker_init_fn=worker_init_fn)
        train_dl = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True,
                              worker_init_fn=worker_init_fn)

        # X_test = torch.Tensor(X_test).permute(0, 3, 1, 2)
        X_test = np.array(X_test)
        X_test = torch.from_numpy(X_test).float().permute(0, 3, 1, 2)

        y_test = torch.Tensor(y_test).to(dtype=torch.long)
        dataset_test = TensorDataset(X_test, y_test)
        test_dl = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=0)

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

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.99), weight_decay=0.0005)

        best_accuracy_for_each_subject = 0
        best_each_subject_pred = []

        max_iter = config.max_iter
        iter_num = 0
        epochs = max_iter // len(train_dl) + 1

        # 2. 创建训练时间估算器
        timer = TrainingTimer(epochs, sub_idx, total_subjects)
        print(f"Starting training for {epochs} epochs...")

        for epoch in range(1, epochs + 1):
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

            val_acc = num_val_correct / num_val_examples
            middle1_val_acc = middle1_num_val_correct / num_val_examples
            middle2_val_acc = middle2_num_val_correct / num_val_examples

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
            torch.save(model.state_dict(), weight_path)

        # 评估结果
        uf1, uar, accuracy = recognition_evaluation(n_subName, y_test.cpu().numpy(), best_each_subject_pred, show=True)
        all_accuracy_dict[n_subName] = [uf1, uar, accuracy]

        total_gt.extend(y_test.cpu().numpy())
        total_pred.extend(best_each_subject_pred)

    # 总体评估
    print("Overall Results:")
    uf1, uar, accuracy = recognition_evaluation("Overall", total_gt, total_pred, show=True)

    print(f"Total time: {time.time() - t:.2f} seconds")
