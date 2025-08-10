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
    
    # 测试次数
    test_runs = 5
    print(f"将进行 {test_runs} 次测试取平均值...")
    
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
    cpu_times = []
    for run in range(test_runs):
        start_time = time.time()
        for _ in range(100):
            a = np.random.rand(1000, 1000)
            b = np.random.rand(1000, 1000)
            c = np.dot(a, b)
        cpu_matrix_time = time.time() - start_time
        cpu_times.append(cpu_matrix_time)
        print(f"  第{run+1}次: {cpu_matrix_time:.3f}秒")
    
    avg_cpu_time = np.mean(cpu_times)
    std_cpu_time = np.std(cpu_times)
    print(f"CPU矩阵运算(100次1000x1000): {avg_cpu_time:.3f}±{std_cpu_time:.3f}秒 (平均值±标准差)")
    
    # 3. 内存性能测试
    print("\n3. 内存性能测试:")
    memory = psutil.virtual_memory()
    print(f"总内存: {memory.total / 1024**3:.1f}GB")
    print(f"可用内存: {memory.available / 1024**3:.1f}GB")
    print(f"内存使用率: {memory.percent}%")
    
    # 内存读写测试
    memory_write_times = []
    memory_read_times = []
    
    for run in range(test_runs):
        # 写入测试
        start_time = time.time()
        large_array = np.random.rand(10000, 10000)
        memory_write_time = time.time() - start_time
        memory_write_times.append(memory_write_time)
        print(f"  第{run+1}次写入: {memory_write_time:.3f}秒")
        
        # 读取测试
        start_time = time.time()
        result = np.sum(large_array)
        memory_read_time = time.time() - start_time
        memory_read_times.append(memory_read_time)
        print(f"  第{run+1}次读取: {memory_read_time:.3f}秒")
    
    avg_memory_write_time = np.mean(memory_write_times)
    avg_memory_read_time = np.mean(memory_read_times)
    std_memory_write_time = np.std(memory_write_times)
    std_memory_read_time = np.std(memory_read_times)
    
    print(f"内存写入(100M数据): {avg_memory_write_time:.3f}±{std_memory_write_time:.3f}秒")
    print(f"内存读取(100M数据): {avg_memory_read_time:.3f}±{std_memory_read_time:.3f}秒")
    
    # 4. 磁盘I/O测试
    print("\n4. 磁盘I/O测试:")
    disk_write_times = []
    disk_read_times = []
    
    for run in range(test_runs):
        # 写入测试
        start_time = time.time()
        test_data = np.random.rand(1000, 1000)
        np.save('temp_test.npy', test_data)
        disk_write_time = time.time() - start_time
        disk_write_times.append(disk_write_time)
        print(f"  第{run+1}次写入: {disk_write_time:.3f}秒")
        
        # 读取测试
        start_time = time.time()
        loaded_data = np.load('temp_test.npy')
        disk_read_time = time.time() - start_time
        disk_read_times.append(disk_read_time)
        print(f"  第{run+1}次读取: {disk_read_time:.3f}秒")
        
        # 清理测试文件
        os.remove('temp_test.npy')
    
    avg_disk_write_time = np.mean(disk_write_times)
    avg_disk_read_time = np.mean(disk_read_times)
    std_disk_write_time = np.std(disk_write_times)
    std_disk_read_time = np.std(disk_read_times)
    
    print(f"磁盘写入(8M数据): {avg_disk_write_time:.3f}±{std_disk_write_time:.3f}秒")
    print(f"磁盘读取(8M数据): {avg_disk_read_time:.3f}±{std_disk_read_time:.3f}秒")
    
    # 5. GPU性能测试
    print("\n5. GPU性能测试:")
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        torch.cuda.set_device(0)
        
        # GPU内存测试
        gpu_memory = torch.cuda.get_device_properties(0).total_memory
        print(f"GPU 0 总内存: {gpu_memory / 1024**3:.1f}GB")
        
        # GPU计算测试
        gpu_compute_times = []
        gpu_memory_times = []
        
        for run in range(test_runs):
            # 计算测试
            start_time = time.time()
            x = torch.randn(1000, 1000).to(device)
            y = torch.randn(1000, 1000).to(device)
            for _ in range(100):
                z = torch.matmul(x, y)
            torch.cuda.synchronize()
            gpu_compute_time = time.time() - start_time
            gpu_compute_times.append(gpu_compute_time)
            print(f"  第{run+1}次计算: {gpu_compute_time:.3f}秒")
            
            # 内存带宽测试
            start_time = time.time()
            large_tensor = torch.randn(5000, 5000).to(device)
            torch.cuda.synchronize()
            gpu_memory_time = time.time() - start_time
            gpu_memory_times.append(gpu_memory_time)
            print(f"  第{run+1}次内存分配: {gpu_memory_time:.3f}秒")
            
            # 清理GPU内存
            del x, y, z, large_tensor
            torch.cuda.empty_cache()
        
        avg_gpu_compute_time = np.mean(gpu_compute_times)
        avg_gpu_memory_time = np.mean(gpu_memory_times)
        std_gpu_compute_time = np.std(gpu_compute_times)
        std_gpu_memory_time = np.std(gpu_memory_times)
        
        print(f"GPU矩阵乘法(100次1000x1000): {avg_gpu_compute_time:.3f}±{std_gpu_compute_time:.3f}秒")
        print(f"GPU内存分配(100M数据): {avg_gpu_memory_time:.3f}±{std_gpu_memory_time:.3f}秒")
    
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
        
        # 存储每个batch size的结果
        batch_results = {}
        
        for batch_size in batch_sizes:
            batch_times = []
            batch_throughputs = []
            
            for run in range(test_runs):
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
                
                batch_times.append(avg_time)
                batch_throughputs.append(throughput)
                print(f"  Batch {batch_size:2d} 第{run+1}次: {avg_time*1000:6.2f}ms, {throughput:8.1f} samples/s")
            
            # 计算平均值和标准差
            avg_time = np.mean(batch_times)
            avg_throughput = np.mean(batch_throughputs)
            std_time = np.std(batch_times)
            std_throughput = np.std(batch_throughputs)
            
            print(f"Batch Size {batch_size:2d}: {avg_time*1000:6.2f}±{std_time*1000:6.2f}ms, {avg_throughput:8.1f}±{std_throughput:6.1f} samples/s")
            
            batch_results[batch_size] = {
                'avg_time': avg_time,
                'avg_throughput': avg_throughput,
                'std_time': std_time,
                'std_throughput': std_throughput
            }
        
        # 清理
        del model, x
        torch.cuda.empty_cache()
    
    # 7. 数据加载测试
    print("\n7. 数据加载测试:")
    # 模拟数据加载
    data_loading_times = []
    
    for run in range(test_runs):
        start_time = time.time()
        for _ in range(100):
            # 模拟图像加载和预处理
            fake_image = np.random.rand(48, 48, 38)
            processed = fake_image / 255.0
            tensor = torch.from_numpy(processed).float().permute(2, 0, 1)
        data_loading_time = time.time() - start_time
        data_loading_times.append(data_loading_time)
        print(f"  第{run+1}次: {data_loading_time:.3f}秒")
    
    avg_data_loading_time = np.mean(data_loading_times)
    std_data_loading_time = np.std(data_loading_times)
    print(f"数据预处理(100次): {avg_data_loading_time:.3f}±{std_data_loading_time:.3f}秒")
    
    # 8. 综合评分
    print("\n8. 综合评分:")
    scores = {}
    
    # CPU评分 (越低越好)
    scores['cpu'] = 1000 / avg_cpu_time
    
    # 内存评分
    scores['memory'] = 1000 / (avg_memory_write_time + avg_memory_read_time)
    
    # 磁盘评分
    scores['disk'] = 1000 / (avg_disk_write_time + avg_disk_read_time)
    
    if torch.cuda.is_available():
        scores['gpu'] = 1000 / avg_gpu_compute_time
        scores['gpu_memory'] = 1000 / avg_gpu_memory_time
    
    scores['data_loading'] = 1000 / avg_data_loading_time
    
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
