import torch
import torch.nn as nn
import time
import numpy as np
import psutil
import os
import platform
from all_model import get_model

def performance_test():
    """性能测试函数"""
    print("=== 系统性能测试 ===")
    
    # 1. 系统信息
    print("\n1. 系统信息:")
    print(f"操作系统: {platform.system()} {platform.release()}")
    print(f"Python版本: {platform.python_version()}")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA版本: {torch.version.cuda}")
        print(f"GPU数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")
    
    # 2. CPU性能测试
    print("\n2. CPU性能测试:")
    # 矩阵运算测试
    start_time = time.time()
    for _ in range(100):
        a = np.random.rand(1000, 1000)
        b = np.random.rand(1000, 1000)
        c = np.dot(a, b)
    cpu_matrix_time = time.time() - start_time
    print(f"CPU矩阵运算(100次1000x1000): {cpu_matrix_time:.3f}秒")
    
    # 3. 内存性能测试
    print("\n3. 内存性能测试:")
    memory = psutil.virtual_memory()
    print(f"总内存: {memory.total / 1024**3:.1f}GB")
    print(f"可用内存: {memory.available / 1024**3:.1f}GB")
    print(f"内存使用率: {memory.percent}%")
    
    # 内存读写测试
    start_time = time.time()
    large_array = np.random.rand(10000, 10000)
    memory_write_time = time.time() - start_time
    print(f"内存写入(100M数据): {memory_write_time:.3f}秒")
    
    start_time = time.time()
    result = np.sum(large_array)
    memory_read_time = time.time() - start_time
    print(f"内存读取(100M数据): {memory_read_time:.3f}秒")
    
    # 4. 磁盘I/O测试
    print("\n4. 磁盘I/O测试:")
    # 写入测试
    start_time = time.time()
    test_data = np.random.rand(1000, 1000)
    np.save('temp_test.npy', test_data)
    disk_write_time = time.time() - start_time
    print(f"磁盘写入(8M数据): {disk_write_time:.3f}秒")
    
    # 读取测试
    start_time = time.time()
    loaded_data = np.load('temp_test.npy')
    disk_read_time = time.time() - start_time
    print(f"磁盘读取(8M数据): {disk_read_time:.3f}秒")
    
    # 清理测试文件
    os.remove('temp_test.npy')
    
    # 5. GPU性能测试
    print("\n5. GPU性能测试:")
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        torch.cuda.set_device(0)
        
        # GPU内存测试
        gpu_memory = torch.cuda.get_device_properties(0).total_memory
        print(f"GPU 0 总内存: {gpu_memory / 1024**3:.1f}GB")
        
        # GPU计算测试
        start_time = time.time()
        x = torch.randn(1000, 1000).to(device)
        y = torch.randn(1000, 1000).to(device)
        for _ in range(100):
            z = torch.matmul(x, y)
        torch.cuda.synchronize()
        gpu_compute_time = time.time() - start_time
        print(f"GPU矩阵乘法(100次1000x1000): {gpu_compute_time:.3f}秒")
        
        # GPU内存带宽测试
        start_time = time.time()
        large_tensor = torch.randn(5000, 5000).to(device)
        torch.cuda.synchronize()
        gpu_memory_time = time.time() - start_time
        print(f"GPU内存分配(100M数据): {gpu_memory_time:.3f}秒")
        
        # 清理GPU内存
        del x, y, z, large_tensor
        torch.cuda.empty_cache()
    
    # 6. 模型性能测试
    print("\n6. 模型性能测试:")
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        torch.cuda.set_device(0)
        
        # 创建模型
        model = get_model("SKD_TSTSAN", 5, 2).to(device)
        model.eval()
        
        # 测试不同batch size的性能
        batch_sizes = [1, 8, 16, 32, 64]
        
        for batch_size in batch_sizes:
            # 创建输入数据
            x = torch.randn(batch_size, 38, 48, 48).to(device)
            
            # 预热
            with torch.no_grad():
                for _ in range(10):
                    _ = model(x)
            
            # 测试性能
            torch.cuda.synchronize()
            start_time = time.time()
            
            with torch.no_grad():
                for _ in range(50):
                    _ = model(x)
            
            torch.cuda.synchronize()
            end_time = time.time()
            
            avg_time = (end_time - start_time) / 50
            throughput = batch_size / avg_time
            
            print(f"Batch Size {batch_size:2d}: {avg_time*1000:6.2f}ms, {throughput:8.1f} samples/s")
        
        # 清理
        del model, x
        torch.cuda.empty_cache()
    
    # 7. 数据加载测试
    print("\n7. 数据加载测试:")
    # 模拟数据加载
    start_time = time.time()
    for _ in range(100):
        # 模拟图像加载和预处理
        fake_image = np.random.rand(48, 48, 38)
        processed = fake_image / 255.0
        tensor = torch.from_numpy(processed).float().permute(2, 0, 1)
    data_loading_time = time.time() - start_time
    print(f"数据预处理(100次): {data_loading_time:.3f}秒")
    
    # 8. 综合评分
    print("\n8. 综合评分:")
    scores = {}
    
    # CPU评分 (越低越好)
    scores['cpu'] = 1000 / cpu_matrix_time
    
    # 内存评分
    scores['memory'] = 1000 / (memory_write_time + memory_read_time)
    
    # 磁盘评分
    scores['disk'] = 1000 / (disk_write_time + disk_read_time)
    
    if torch.cuda.is_available():
        scores['gpu'] = 1000 / gpu_compute_time
        scores['gpu_memory'] = 1000 / gpu_memory_time
    
    scores['data_loading'] = 1000 / data_loading_time
    
    print("各组件性能评分 (越高越好):")
    for component, score in scores.items():
        print(f"{component:15s}: {score:.1f}")
    
    # 总体评分
    overall_score = sum(scores.values()) / len(scores)
    print(f"\n总体性能评分: {overall_score:.1f}")
    
    print("\n=== 测试完成 ===")
    
    return scores

if __name__ == "__main__":
    performance_test()
