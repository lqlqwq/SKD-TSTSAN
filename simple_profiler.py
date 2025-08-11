# simple_profiler.py - 简单的性能分析工具
import time
import json
from collections import defaultdict
import torch

class SimpleProfiler:
    """简单的性能分析器，可以直接集成到训练代码中"""
    
    def __init__(self):
        self.timings = defaultdict(list)
        self.start_times = {}
        self.memory_stats = {}
    
    def start(self, phase_name):
        """开始计时一个阶段"""
        torch.cuda.synchronize()
        self.start_times[phase_name] = time.perf_counter()
    
    def end(self, phase_name):
        """结束计时一个阶段"""
        torch.cuda.synchronize()
        if phase_name in self.start_times:
            elapsed = (time.perf_counter() - self.start_times[phase_name]) * 1000.0  # 转换为毫秒
            self.timings[phase_name].append(elapsed)
            del self.start_times[phase_name]
    
    def record_memory(self, phase_name):
        """记录内存使用情况"""
        self.memory_stats[phase_name] = {
            "allocated_GB": torch.cuda.memory_allocated() / 1e9,
            "reserved_GB": torch.cuda.memory_reserved() / 1e9,
            "max_allocated_GB": torch.cuda.max_memory_allocated() / 1e9,
            "max_reserved_GB": torch.cuda.max_memory_reserved() / 1e9,
        }
    
    def get_summary(self):
        """获取性能统计摘要"""
        summary = {}
        for phase, times in self.timings.items():
            if times:
                avg_time = sum(times) / len(times)
                min_time = min(times)
                max_time = max(times)
                summary[phase] = {
                    "avg_ms": avg_time,
                    "min_ms": min_time,
                    "max_ms": max_time,
                    "count": len(times),
                    "total_ms": sum(times)
                }
        return summary
    
    def print_summary(self):
        """打印性能摘要"""
        summary = self.get_summary()
        
        print("\n" + "="*60)
        print("性能分析摘要")
        print("="*60)
        
        for phase, stats in summary.items():
            print(f"{phase:20s}: 平均 {stats['avg_ms']:8.2f}ms, "
                  f"最小 {stats['min_ms']:8.2f}ms, "
                  f"最大 {stats['max_ms']:8.2f}ms, "
                  f"次数 {stats['count']}")
        
        # 计算总时间
        total_time = sum(stats['total_ms'] for stats in summary.values())
        print(f"\n总耗时: {total_time:.2f}ms")
        
        # 计算各阶段占比
        print("\n各阶段占比:")
        for phase, stats in summary.items():
            percentage = (stats['total_ms'] / total_time) * 100
            print(f"{phase:20s}: {percentage:6.1f}%")
        
        # 内存使用情况
        if self.memory_stats:
            print(f"\n内存使用情况:")
            for phase, mem in self.memory_stats.items():
                print(f"{phase:20s}: 分配 {mem['allocated_GB']:.3f}GB, "
                      f"保留 {mem['reserved_GB']:.3f}GB")
    
    def save_report(self, filename="performance_report.json"):
        """保存详细报告到文件"""
        report = {
            "timings": self.get_summary(),
            "memory_stats": self.memory_stats,
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else "N/A"
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"\n详细报告已保存到: {filename}")

# 使用示例函数
def profile_training_step(profiler, model, optimizer, loss_fn, batch, use_amp=True):
    """分析单个训练步骤的性能"""
    
    x, y = batch[0], batch[1]
    
    # 清零梯度
    profiler.start("zero_grad")
    optimizer.zero_grad(set_to_none=True)
    profiler.end("zero_grad")
    
    if use_amp:
        # 前向传播 (混合精度)
        profiler.start("forward_pass")
        with torch.cuda.amp.autocast(dtype=torch.float16):
            out = model(x)
        profiler.end("forward_pass")
        
        # 损失计算
        profiler.start("loss_computation")
        loss = loss_fn(out, y)
        profiler.end("loss_computation")
        
        # 反向传播
        profiler.start("backward_pass")
        scaler = torch.cuda.amp.GradScaler()
        scaler.scale(loss).backward()
        profiler.end("backward_pass")
        
        # 优化器步骤
        profiler.start("optimizer_step")
        scaler.step(optimizer)
        scaler.update()
        profiler.end("optimizer_step")
    else:
        # 前向传播 (全精度)
        profiler.start("forward_pass")
        out = model(x)
        profiler.end("forward_pass")
        
        # 损失计算
        profiler.start("loss_computation")
        loss = loss_fn(out, y)
        profiler.end("loss_computation")
        
        # 反向传播
        profiler.start("backward_pass")
        loss.backward()
        profiler.end("backward_pass")
        
        # 优化器步骤
        profiler.start("optimizer_step")
        optimizer.step()
        profiler.end("optimizer_step")
    
    # 记录内存使用
    profiler.record_memory("after_step")

# 集成到现有训练代码的示例
def integrate_with_training():
    """展示如何集成到现有训练代码中"""
    
    # 创建分析器
    profiler = SimpleProfiler()
    
    # 在训练循环中使用
    for epoch in range(num_epochs):
        for batch_idx, batch in enumerate(train_loader):
            
            # 数据加载计时
            profiler.start("data_loading")
            # ... 你的数据加载代码 ...
            profiler.end("data_loading")
            
            # CPU到GPU传输计时
            profiler.start("cpu_to_gpu")
            batch = batch.to(device)
            profiler.end("cpu_to_gpu")
            
            # 训练步骤计时
            profile_training_step(profiler, model, optimizer, loss_fn, batch, use_amp=True)
            
            # 每100步打印一次摘要
            if batch_idx % 100 == 0:
                profiler.print_summary()
    
    # 训练结束后保存完整报告
    profiler.save_report("final_training_report.json")

if __name__ == "__main__":
    print("这是一个性能分析工具，可以集成到你的训练代码中。")
    print("使用方法:")
    print("1. 创建 SimpleProfiler() 实例")
    print("2. 在关键代码段前后调用 profiler.start() 和 profiler.end()")
    print("3. 定期调用 profiler.print_summary() 查看性能")
    print("4. 训练结束后调用 profiler.save_report() 保存详细报告")
