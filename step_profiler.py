# step_profiler.py - SKD-TSTSAN项目性能分析工具
import time
import json
import contextlib
from collections import defaultdict
import torch
import torch.nn as nn
import cv2
import numpy as np
from torch.utils.data import DataLoader, Dataset
import os

class StepTimer:
    def __init__(self, sync_cuda=True):
        self.sync = sync_cuda
        self.stats = defaultdict(list)

    @contextlib.contextmanager
    def phase(self, name):
        if self.sync: 
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        yield
        if self.sync: 
            torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) * 1000.0
        self.stats[name].append(dt)

    def summary(self):
        def agg(v):
            if not v:
                return {"avg_ms": 0, "p95_ms": 0, "n": 0}
            v_sorted = sorted(v)
            p95 = v_sorted[int(max(0, len(v) * 0.95 - 1))]
            return {"avg_ms": sum(v)/len(v), "p95_ms": p95, "n": len(v)}
        return {k: agg(v) for k, v in self.stats.items()}

def to_device(batch, device="cuda", non_blocking=True):
    if isinstance(batch, (list, tuple)):
        return [to_device(x, device, non_blocking) for x in batch]
    if isinstance(batch, dict):
        return {k: to_device(v, device, non_blocking) for k,v in batch.items()}
    if torch.is_tensor(batch):
        return batch.to(device, non_blocking=non_blocking)
    return batch

def normalize_gray(img):
    """图像归一化函数"""
    return img.astype(np.float32) / 255.0

def load_and_preprocess_sample(data_path, subName, expName, caseName):
    """加载和预处理单个样本"""
    # 构建文件路径
    apex_path = os.path.join(data_path, subName, expName, caseName, f"{caseName}_apex.jpg")
    onset_path = os.path.join(data_path, subName, expName, caseName, f"{caseName}_onset.jpg")
    u1_path = os.path.join(data_path, subName, expName, caseName, f"{caseName}_1_u.jpg")
    v1_path = os.path.join(data_path, subName, expName, caseName, f"{caseName}_1_v.jpg")
    u2_path = os.path.join(data_path, subName, expName, caseName, f"{caseName}_2_u.jpg")
    v2_path = os.path.join(data_path, subName, expName, caseName, f"{caseName}_2_v.jpg")
    
    # 读取图像
    apex = cv2.imread(apex_path, cv2.IMREAD_GRAYSCALE)
    onset = cv2.imread(onset_path, cv2.IMREAD_GRAYSCALE)
    u1 = cv2.imread(u1_path, cv2.IMREAD_GRAYSCALE)
    v1 = cv2.imread(v1_path, cv2.IMREAD_GRAYSCALE)
    u2 = cv2.imread(u2_path, cv2.IMREAD_GRAYSCALE)
    v2 = cv2.imread(v2_path, cv2.IMREAD_GRAYSCALE)
    
    # 调整大小
    apex = cv2.resize(apex, (48, 48))
    onset = cv2.resize(onset, (48, 48))
    u1 = cv2.resize(u1, (48, 48))
    v1 = cv2.resize(v1, (48, 48))
    u2 = cv2.resize(u2, (48, 48))
    v2 = cv2.resize(v2, (48, 48))
    
    # 归一化
    apex = normalize_gray(apex)
    onset = normalize_gray(onset)
    u1 = normalize_gray(u1)
    v1 = normalize_gray(v1)
    u2 = normalize_gray(u2)
    v2 = normalize_gray(v2)
    
    # 构建38通道输入
    # 前16通道: apex (16通道)
    # 第17通道: onset (1通道)
    # 第18-21通道: u1, v1, u2, v2 (4通道)
    # 第22-38通道: 其他特征 (17通道，这里用零填充)
    
    input_data = np.zeros((38, 48, 48), dtype=np.float32)
    
    # 填充apex (16通道)
    for i in range(16):
        input_data[i] = apex
    
    # 填充onset (1通道)
    input_data[16] = onset
    
    # 填充u1, v1, u2, v2 (4通道)
    input_data[17] = u1
    input_data[18] = v1
    input_data[19] = u2
    input_data[20] = v2
    
    return input_data

class MockDataset(Dataset):
    """模拟数据集用于性能测试"""
    def __init__(self, data_path, num_samples=100):
        self.data_path = data_path
        self.num_samples = num_samples
        self.samples = []
        
        # 生成模拟样本列表
        subNames = ['sub01', 'sub02', 'sub03']
        expNames = ['exp01', 'exp02', 'exp03']
        caseNames = ['case01', 'case02', 'case03']
        
        for i in range(num_samples):
            sub = subNames[i % len(subNames)]
            exp = expNames[i % len(expNames)]
            case = caseNames[i % len(caseNames)]
            self.samples.append((sub, exp, case))
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        sub, exp, case = self.samples[idx]
        
        # 模拟数据加载时间
        time.sleep(0.001)  # 1ms模拟磁盘读取
        
        # 生成随机数据代替实际图像
        input_data = np.random.rand(38, 48, 48).astype(np.float32)
        label = np.random.randint(0, 7)  # 7个表情类别
        
        return torch.from_numpy(input_data), torch.tensor(label, dtype=torch.long)

class MockModel(nn.Module):
    """模拟SKD-TSTSAN模型用于性能测试"""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(38, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, 7)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def profile_training_steps(data_path, steps=50, batch_size=32, device="cuda", 
                          use_amp=True, save_json="training_profile.json"):
    """分析训练步骤的性能"""
    
    # 设置设备
    torch.cuda.set_device(0)
    device = torch.device(device)
    
    # 创建模拟数据集和数据加载器
    dataset = MockDataset(data_path, num_samples=steps * batch_size)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=0,  # 单线程便于分析
        pin_memory=True
    )
    
    # 创建模型和优化器
    model = MockModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()
    
    # 启用性能优化
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    
    # 创建计时器
    timer = StepTimer(sync_cuda=True)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    
    # 重置内存统计
    torch.cuda.reset_peak_memory_stats()
    
    model.train()
    it = iter(dataloader)
    seen = 0
    
    print(f"开始性能分析，运行 {steps} 步...")
    
    for step in range(steps):
        if step % 10 == 0:
            print(f"进度: {step}/{steps}")
        
        # 数据加载阶段
        with timer.phase("data_loading"):
            batch = next(it)
        
        # CPU到GPU传输阶段
        with timer.phase("cpu_to_gpu"):
            batch = to_device(batch, device=device, non_blocking=True)
        
        # 解包数据
        x, y = batch[0], batch[1]
        
        # 清零梯度
        with timer.phase("zero_grad"):
            optimizer.zero_grad(set_to_none=True)
        
        if use_amp:
            # 前向传播 (混合精度)
            with timer.phase("forward_pass"):
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    out = model(x)
            
            # 损失计算
            with timer.phase("loss_computation"):
                loss = loss_fn(out, y)
            
            # 反向传播
            with timer.phase("backward_pass"):
                scaler.scale(loss).backward()
            
            # 优化器步骤
            with timer.phase("optimizer_step"):
                scaler.step(optimizer)
                scaler.update()
        else:
            # 前向传播 (全精度)
            with timer.phase("forward_pass"):
                out = model(x)
            
            # 损失计算
            with timer.phase("loss_computation"):
                loss = loss_fn(out, y)
            
            # 反向传播
            with timer.phase("backward_pass"):
                loss.backward()
            
            # 优化器步骤
            with timer.phase("optimizer_step"):
                optimizer.step()
        
        if hasattr(x, "shape"):
            seen += x.shape[0]
    
    # 收集内存统计
    mem = {
        "allocated_GB": torch.cuda.memory_allocated()/1e9,
        "reserved_GB": torch.cuda.memory_reserved()/1e9,
        "peak_allocated_GB": torch.cuda.max_memory_allocated()/1e9,
        "peak_reserved_GB": torch.cuda.max_memory_reserved()/1e9,
    }
    
    # 生成报告
    report = {
        "device": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "allow_tf32": {
            "matmul": torch.backends.cuda.matmul.allow_tf32,
            "cudnn": torch.backends.cudnn.allow_tf32,
        },
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "batch_size": batch_size,
        "steps": steps,
        "samples_seen": seen,
        "use_amp": use_amp,
        "phase_timing_ms": timer.summary(),
        "memory": mem,
        "throughput": {
            "samples_per_second": seen / (sum(timer.stats["data_loading"]) / 1000.0),
            "steps_per_second": steps / (sum(timer.stats["data_loading"]) / 1000.0),
        }
    }
    
    # 保存报告
    with open(save_json, "w", encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print(f"\n=== 性能分析报告 ===")
    print(f"设备: {report['device']}")
    print(f"批次大小: {batch_size}")
    print(f"总步数: {steps}")
    print(f"处理样本数: {seen}")
    print(f"混合精度: {use_amp}")
    
    print(f"\n=== 各阶段耗时 (毫秒) ===")
    for phase, stats in report["phase_timing_ms"].items():
        print(f"{phase:15s}: 平均 {stats['avg_ms']:8.2f}ms, P95 {stats['p95_ms']:8.2f}ms, 次数 {stats['n']}")
    
    print(f"\n=== 内存使用 (GB) ===")
    for key, value in report["memory"].items():
        print(f"{key:20s}: {value:.3f}")
    
    print(f"\n=== 吞吐量 ===")
    print(f"样本/秒: {report['throughput']['samples_per_second']:.1f}")
    print(f"步数/秒: {report['throughput']['steps_per_second']:.1f}")
    
    print(f"\n[完成] 详细报告已保存到 {save_json}")
    
    return report

if __name__ == "__main__":
    # 设置数据路径 (请根据实际情况修改)
    data_path = "/path/to/your/data"
    
    # 运行性能分析
    profile_training_steps(
        data_path=data_path,
        steps=50,
        batch_size=32,
        device="cuda",
        use_amp=True,
        save_json="training_profile.json"
    )
