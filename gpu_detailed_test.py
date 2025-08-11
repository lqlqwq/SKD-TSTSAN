import torch
import time
import numpy as np

def gpu_detailed_test():
    """详细的GPU性能诊断"""
    print("=== 详细GPU性能诊断 ===")
    
    if not torch.cuda.is_available():
        print("CUDA不可用")
        return
    
    device = torch.device('cuda:0')
    torch.cuda.set_device(0)
    
    # 1. GPU基本信息
    print(f"\n1. GPU基本信息:")
    print(f"GPU名称: {torch.cuda.get_device_name(0)}")
    print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    print(f"CUDA版本: {torch.version.cuda}")
    print(f"cuDNN版本: {torch.backends.cudnn.version()}")
    
    # 2. GPU计算性能测试
    print(f"\n2. GPU计算性能测试:")
    
    # 小矩阵测试
    sizes = [100, 500, 1000, 2000]
    for size in sizes:
        times = []
        for _ in range(10):
            x = torch.randn(size, size).to(device)
            y = torch.randn(size, size).to(device)
            
            torch.cuda.synchronize()
            start = time.time()
            
            for _ in range(50):
                z = torch.matmul(x, y)
            
            torch.cuda.synchronize()
            end = time.time()
            times.append((end - start) / 50)
        
        avg_time = np.mean(times)
        print(f"  {size}x{size} 矩阵乘法: {avg_time*1000:.2f}ms")
    
    # 3. GPU内存带宽测试
    print(f"\n3. GPU内存带宽测试:")
    
    sizes = [1000, 2000, 5000, 10000]
    for size in sizes:
        # 写入测试
        torch.cuda.synchronize()
        start = time.time()
        x = torch.randn(size, size).to(device)
        torch.cuda.synchronize()
        write_time = time.time() - start
        
        # 读取测试
        torch.cuda.synchronize()
        start = time.time()
        result = torch.sum(x)
        torch.cuda.synchronize()
        read_time = time.time() - start
        
        data_size = size * size * 4  # float32 = 4 bytes
        write_bandwidth = data_size / write_time / 1024**3  # GB/s
        read_bandwidth = data_size / read_time / 1024**3   # GB/s
        
        print(f"  {size}x{size} 数据 ({data_size/1024**2:.1f}MB):")
        print(f"    写入带宽: {write_bandwidth:.1f} GB/s")
        print(f"    读取带宽: {read_bandwidth:.1f} GB/s")
        
        del x
    
    # 4. CPU-GPU数据传输测试
    print(f"\n4. CPU-GPU数据传输测试:")
    
    sizes = [1000, 2000, 5000]
    for size in sizes:
        # CPU到GPU
        cpu_tensor = torch.randn(size, size)
        
        torch.cuda.synchronize()
        start = time.time()
        gpu_tensor = cpu_tensor.to(device)
        torch.cuda.synchronize()
        cpu_to_gpu_time = time.time() - start
        
        # GPU到CPU
        torch.cuda.synchronize()
        start = time.time()
        cpu_tensor_back = gpu_tensor.cpu()
        torch.cuda.synchronize()
        gpu_to_cpu_time = time.time() - start
        
        data_size = size * size * 4
        cpu_to_gpu_bandwidth = data_size / cpu_to_gpu_time / 1024**3
        gpu_to_cpu_bandwidth = data_size / gpu_to_cpu_time / 1024**3
        
        print(f"  {size}x{size} 数据 ({data_size/1024**2:.1f}MB):")
        print(f"    CPU→GPU: {cpu_to_gpu_bandwidth:.1f} GB/s")
        print(f"    GPU→CPU: {gpu_to_cpu_bandwidth:.1f} GB/s")
    
    # 5. GPU利用率测试
    print(f"\n5. GPU利用率测试:")
    
    # 创建计算密集型任务
    x = torch.randn(2000, 2000).to(device)
    y = torch.randn(2000, 2000).to(device)
    
    print("  开始持续计算测试 (10秒)...")
    torch.cuda.synchronize()
    start = time.time()
    
    iterations = 0
    while time.time() - start < 10:
        z = torch.matmul(x, y)
        iterations += 1
    
    torch.cuda.synchronize()
    end = time.time()
    
    ops_per_second = iterations / (end - start)
    print(f"  计算速度: {ops_per_second:.1f} 矩阵乘法/秒")
    
    # 6. 内存碎片测试
    print(f"\n6. GPU内存碎片测试:")
    
    # 分配和释放不同大小的内存
    torch.cuda.empty_cache()
    initial_memory = torch.cuda.memory_allocated()
    
    tensors = []
    for i in range(10):
        size = 1000 + i * 100
        tensor = torch.randn(size, size).to(device)
        tensors.append(tensor)
    
    after_alloc = torch.cuda.memory_allocated()
    
    # 随机释放一些tensor
    for i in [1, 3, 5, 7]:
        if i < len(tensors):
            del tensors[i]
    
    torch.cuda.empty_cache()
    after_free = torch.cuda.memory_allocated()
    
    print(f"  初始内存: {initial_memory / 1024**2:.1f} MB")
    print(f"  分配后内存: {after_alloc / 1024**2:.1f} MB")
    print(f"  释放后内存: {after_free / 1024**2:.1f} MB")
    
    # 清理
    del tensors, x, y
    torch.cuda.empty_cache()

if __name__ == "__main__":
    gpu_detailed_test()
