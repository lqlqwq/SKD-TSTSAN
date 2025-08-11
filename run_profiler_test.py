#!/usr/bin/env python3
# run_profiler_test.py - 直接运行性能测试

import torch
import torch.nn as nn
import numpy as np
import time
from simple_profiler import SimpleProfiler, profile_training_step

def create_mock_model():
    """创建模拟的SKD-TSTSAN模型"""
    class MockSKDModel(nn.Module):
        def __init__(self):
            super().__init__()
            # 模拟SKD-TSTSAN的复杂度
            self.conv1 = nn.Conv2d(38, 64, 3, padding=1)
            self.bn1 = nn.BatchNorm2d(64)
            self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
            self.bn2 = nn.BatchNorm2d(128)
            self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
            self.bn3 = nn.BatchNorm2d(256)
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc1 = nn.Linear(256, 128)
            self.fc2 = nn.Linear(128, 7)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(0.5)
        
        def forward(self, x):
            x = self.relu(self.bn1(self.conv1(x)))
            x = self.relu(self.bn2(self.conv2(x)))
            x = self.relu(self.bn3(self.conv3(x)))
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            x = self.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2(x)
            return x
    
    return MockSKDModel()

def create_mock_data(batch_size=32):
    """创建模拟数据"""
    # 模拟38通道输入数据
    x = torch.randn(batch_size, 38, 48, 48, dtype=torch.float32)
    y = torch.randint(0, 7, (batch_size,), dtype=torch.long)
    return x, y

def run_performance_test():
    """运行性能测试"""
    
    print("="*60)
    print("SKD-TSTSAN 训练性能分析")
    print("="*60)
    
    # 设置设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(0)
    
    print(f"使用设备: {device}")
    print(f"GPU名称: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA版本: {torch.version.cuda}")
    
    # 启用性能优化
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    
    # 创建模型和优化器
    model = create_mock_model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()
    
    # 创建性能分析器
    profiler = SimpleProfiler()
    
    # 测试参数
    batch_sizes = [16, 32, 64, 128, 256]
    steps_per_test = 50
    
    print(f"\n开始性能测试，每个批次大小测试 {steps_per_test} 步...")
    
    for batch_size in batch_sizes:
        print(f"\n--- 测试批次大小: {batch_size} ---")
        
        # 重置分析器
        profiler = SimpleProfiler()
        
        # 预热
        print("预热中...")
        for _ in range(10):
            x, y = create_mock_data(batch_size)
            x, y = x.to(device), y.to(device)
            profile_training_step(profiler, model, optimizer, loss_fn, (x, y), use_amp=True)
        
        # 正式测试
        print("正式测试中...")
        for step in range(steps_per_test):
            if step % 10 == 0:
                print(f"  进度: {step}/{steps_per_test}")
            
            # 数据加载模拟
            profiler.start("data_loading")
            x, y = create_mock_data(batch_size)
            profiler.end("data_loading")
            
            # CPU到GPU传输
            profiler.start("cpu_to_gpu")
            x, y = x.to(device), y.to(device)
            profiler.end("cpu_to_gpu")
            
            # 训练步骤
            profile_training_step(profiler, model, optimizer, loss_fn, (x, y), use_amp=True)
        
        # 打印结果
        profiler.print_summary()
        
        # 计算吞吐量
        summary = profiler.get_summary()
        total_time = sum(stats['total_ms'] for stats in summary.values())
        samples_per_second = (batch_size * steps_per_test) / (total_time / 1000.0)
        
        print(f"吞吐量: {samples_per_second:.1f} 样本/秒")
        print(f"平均每步时间: {total_time / steps_per_test:.2f}ms")
    
    # 保存最终报告
    profiler.save_report(f"skd_tstsan_performance_report.json")
    
    print(f"\n" + "="*60)
    print("性能测试完成！")
    print("="*60)

def run_memory_test():
    """运行内存使用测试"""
    
    print("\n" + "="*60)
    print("内存使用测试")
    print("="*60)
    
    device = torch.device("cuda:0")
    torch.cuda.set_device(0)
    
    # 重置内存统计
    torch.cuda.reset_peak_memory_stats()
    
    model = create_mock_model().to(device)
    
    batch_sizes = [16, 32, 64, 128, 256, 512, 1024]
    
    print("测试不同批次大小的内存使用:")
    print(f"{'批次大小':<10} {'输入数据(MB)':<15} {'模型参数(MB)':<15} {'总显存(GB)':<15} {'利用率':<10}")
    print("-" * 70)
    
    for batch_size in batch_sizes:
        # 计算输入数据大小
        input_size_mb = (batch_size * 38 * 48 * 48 * 4) / (1024 * 1024)  # float32 = 4 bytes
        
        # 计算模型参数大小
        total_params = sum(p.numel() for p in model.parameters())
        model_size_mb = (total_params * 4) / (1024 * 1024)  # float32 = 4 bytes
        
        # 创建数据并测量显存
        x = torch.randn(batch_size, 38, 48, 48, dtype=torch.float32).to(device)
        y = torch.randint(0, 7, (batch_size,), dtype=torch.long).to(device)
        
        # 前向传播
        with torch.no_grad():
            _ = model(x)
        
        # 获取显存使用
        allocated_gb = torch.cuda.memory_allocated() / 1e9
        total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        utilization = (allocated_gb / total_gb) * 100
        
        print(f"{batch_size:<10} {input_size_mb:<15.1f} {model_size_mb:<15.1f} {allocated_gb:<15.3f} {utilization:<10.1f}%")
        
        # 清理
        del x, y
        torch.cuda.empty_cache()
    
    print(f"\nGPU总显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

if __name__ == "__main__":
    try:
        # 运行性能测试
        run_performance_test()
        
        # 运行内存测试
        run_memory_test()
        
    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
