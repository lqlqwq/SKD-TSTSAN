#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
通用 GPU 训练相关性能基准：
- 设备与软件环境信息
- 显存内部带宽 (D2D)
- PCIe 带宽：H2D / D2H（pageable vs pinned）
- GEMM 吞吐：FP32(TF32开/关) / FP16 / BF16
- 数据加载与上卡吞吐（合成数据，模拟多帧图像批）
- （可选）单机多卡 NCCL All-Reduce 带宽与延迟

用法：
  单机单卡：
    python perf_bench.py

  自定义参数：
    python perf_bench.py --iters 50 --warmup 10 --sizes 2048 4096 8192 --batch 16 --frames 8 --height 224 --width 224 --workers 8

  多卡通信（单机，按需）：
    torchrun --standalone --nproc_per_node=4 perf_bench.py --ddp

运行后会在当前目录生成 JSON 报告文件：perf_report_<timestamp>.json
"""
import argparse, json, os, sys, time, platform, datetime
from statistics import mean
import torch
from torch.utils.data import Dataset, DataLoader

# ---------- 工具: CUDA 事件计时 ----------
def cuda_time(fn, iters=50, warmup=10, sync=True):
    assert torch.cuda.is_available()
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    # 预热
    for _ in range(warmup):
        fn()
    if sync: torch.cuda.synchronize()
    # 正式计时
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    if sync: torch.cuda.synchronize()
    ms = start.elapsed_time(end) / iters
    return ms

# ---------- 显示/采集环境信息 ----------
def collect_env():
    dev = torch.cuda.get_device_properties(0)
    cap = torch.cuda.get_device_capability(0)
    env = {
        "timestamp": datetime.datetime.now().isoformat(),
        "hostname": platform.node(),
        "os": f"{platform.system()} {platform.release()}",
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "cuda_toolkit": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu_name": dev.name,
        "gpu_total_mem_GB": round(dev.total_memory / (1024**3), 2),
        "sm_capability": f"{cap[0]}.{cap[1]}",
        "multi_gpu_count": torch.cuda.device_count(),
        "driver": None,
    }
    try:
        # 仅供参考：PyTorch 不暴露驱动版，这里尝试从 nvidia-smi
        import subprocess
        out = subprocess.check_output(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True)
        env["driver"] = out.strip().splitlines()[0]
    except Exception:
        pass
    return env

# ---------- 带宽测试 ----------
def test_bandwidth(args):
    device = torch.device("cuda:0")
    N = args.band_bytes // 4  # float32
    results = {}

    # D2D（显存内部复制）
    x = torch.empty(N, dtype=torch.float32, device=device)
    y = torch.empty_like(x)
    ms = cuda_time(lambda: y.copy_(x, non_blocking=True), args.iters, args.warmup)
    results["D2D_GBps"] = (x.nbytes / 1e9) / (ms / 1e3)

    # H2D/D2H pageable
    cpu_page = torch.empty(N, dtype=torch.float32)  # 非 pinned
    ms_h2d = cuda_time(lambda: x.copy_(cpu_page, non_blocking=True), args.iters, args.warmup)
    ms_d2h = cuda_time(lambda: cpu_page.copy_(x, non_blocking=True), args.iters, args.warmup)
    results["H2D_pageable_GBps"] = (x.nbytes / 1e9) / (ms_h2d / 1e3)
    results["D2H_pageable_GBps"] = (x.nbytes / 1e9) / (ms_d2h / 1e3)

    # H2D/D2H pinned
    cpu_pin = torch.empty(N, dtype=torch.float32, pin_memory=True)
    ms_h2d = cuda_time(lambda: x.copy_(cpu_pin, non_blocking=True), args.iters, args.warmup)
    ms_d2h = cuda_time(lambda: cpu_pin.copy_(x, non_blocking=True), args.iters, args.warmup)
    results["H2D_pinned_GBps"] = (x.nbytes / 1e9) / (ms_h2d / 1e3)
    results["D2H_pinned_GBps"] = (x.nbytes / 1e9) / (ms_d2h / 1e3)

    return results

# ---------- GEMM 吞吐 ----------
def gemm_once(n, dtype, use_tf32):
    # 设置 TF32
    torch.backends.cuda.matmul.allow_tf32 = bool(use_tf32)
    if dtype == torch.float32 and not use_tf32:
        torch.set_float32_matmul_precision('highest')
    else:
        torch.set_float32_matmul_precision('high')
    device = "cuda:0"
    a = torch.randn((n, n), device=device, dtype=dtype)
    b = torch.randn((n, n), device=device, dtype=dtype)

    def fn():
        # 复用 out 避免重复分配
        torch.matmul(a, b, out=a)

    ms = cuda_time(fn, iters=20, warmup=5)
    # 理论 FLOPs: 2*n^3
    tflops = (2 * (n**3) / ms * 1e-9)
    return {"ms": round(ms, 4), "TFLOPs": round(tflops, 2)}

def test_gemm(args):
    sizes = args.sizes
    results = {"FP32_TF32_on": {}, "FP32_TF32_off": {}, "FP16": {}, "BF16": {}}
    for n in sizes:
        results["FP32_TF32_on"][n]  = gemm_once(n, torch.float32, True)
        results["FP32_TF32_off"][n] = gemm_once(n, torch.float32, False)
        # autocast 对 matmul 通常已走 tensor core
        for dt, key in [(torch.float16, "FP16"), (torch.bfloat16, "BF16")]:
            try:
                results[key][n] = gemm_once(n, dt, True)
            except RuntimeError as e:
                results[key][n] = {"error": str(e)}
    return results

# ---------- 数据加载与上卡吞吐（合成数据） ----------
class SyntheticVideo(Dataset):
    def __init__(self, length, frames, c, h, w, dtype=torch.float32):
        self.length, self.frames, self.c, self.h, self.w, self.dtype = length, frames, c, h, w, dtype
    def __len__(self): return self.length
    def __getitem__(self, idx):
        # 生成在 CPU，模拟解码后张量（随机数据）
        x = torch.randn(self.frames, self.c, self.h, self.w, dtype=self.dtype)
        y = torch.randint(0, 10, (1,), dtype=torch.long)[0]
        return x, y

def collate_pad(batch):
    xs, ys = zip(*batch)
    return torch.stack(xs, dim=0), torch.tensor(ys)

def test_input_pipeline(args):
    device = torch.device("cuda:0")
    dataset = SyntheticVideo(
        length=args.loader_samples,
        frames=args.frames,
        c=3,
        h=args.height,
        w=args.width,
        dtype=torch.float32,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=args.pin_memory,
        persistent_workers=True if args.workers > 0 else False,
        prefetch_factor=args.prefetch if args.workers > 0 else None,
        collate_fn=collate_pad,
        drop_last=True,
    )

    # 只测数据从 DataLoader -> GPU 的吞吐与上卡时间
    steps = min(args.loader_steps, len(loader))
    load_times, h2d_times = [], []
    it = iter(loader)
    # 预热
    for _ in range(3):
        x, y = next(it)
        if args.pin_memory:
            x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
        else:
            x = x.to(device); y = y.to(device)
        torch.cuda.synchronize()

    # 正式
    it = iter(loader)
    for _ in range(steps):
        t0 = time.perf_counter()
        x, y = next(it)
        t1 = time.perf_counter()
        if args.pin_memory:
            torch.cuda.synchronize()
            s0 = time.perf_counter()
            x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
            torch.cuda.synchronize()
            s1 = time.perf_counter()
        else:
            s0 = time.perf_counter()
            x = x.to(device); y = y.to(device)
            torch.cuda.synchronize()
            s1 = time.perf_counter()
        load_times.append(t1 - t0)
        h2d_times.append(s1 - s0)

    # 统计
    batch_elems = args.batch * args.frames * 3 * args.height * args.width
    bytes_per_batch = batch_elems * 4  # float32
    h2d_gbps = (bytes_per_batch / 1e9) / mean(h2d_times)
    items_per_s = args.batch / mean([lt + ht for lt, ht in zip(load_times, h2d_times)])

    return {
        "cpu_batch_load_ms": round(mean(load_times) * 1000, 3),
        "h2d_per_batch_ms": round(mean(h2d_times) * 1000, 3),
        "estimated_h2d_GBps": round(h2d_gbps, 2),
        "samples_per_second": round(items_per_s, 2),
        "shape": [args.batch, args.frames, 3, args.height, args.width],
        "num_workers": args.workers,
        "pin_memory": args.pin_memory,
        "prefetch_factor": args.prefetch,
    }

# ---------- 多卡 NCCL All-Reduce（可选） ----------
def ddp_allreduce_bench(args):
    # 在 torchrun 下，由 torch.distributed 初始化
    import torch.distributed as dist
    if not dist.is_available(): return {"error": "torch.distributed not available"}
    if not dist.is_initialized():
        return {"error": "run with torchrun and --ddp to enable"}

    world = dist.get_world_size()
    rank  = dist.get_rank()
    torch.cuda.set_device(rank % torch.cuda.device_count())
    device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")

    sizes = [16<<20, 32<<20, 64<<20, 128<<20]  # 16MB~128MB
    out = {}
    for sz in sizes:
        numel = sz // 4
        t = torch.randn(numel, device=device, dtype=torch.float32)
        # 预热
        for _ in range(5):
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize()
        # 计时
        start = torch.cuda.Event(True); end = torch.cuda.Event(True)
        iters = 30
        start.record()
        for _ in range(iters):
            dist.all_reduce(t, op=dist.ReduceOp.SUM, async_op=False)
        end.record(); torch.cuda.synchronize()
        ms = start.elapsed_time(end) / iters
        # 每次 all-reduce 理论传输量（近似）：2*(world-1)/world * size
        # 参见环式 all-reduce 模型
        eff = 2.0 * (world - 1) / world * (sz / 1e9) / (ms / 1e3)  # GB/s
        out[f"{sz//(1<<20)}MB"] = {"latency_ms": round(ms, 3), "eff_bandwidth_GBps": round(eff, 2)}
    # 只在 rank0 打印/返回
    return out

# ---------- 主程序 ----------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--band-bytes", dest="band_bytes", type=int, default=40_000_000, help="带宽测试的数据大小（字节）")
    p.add_argument("--sizes", type=int, nargs="+", default=[2048, 4096, 8192], help="GEMM 测试边长")
    # DataLoader
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--frames", type=int, default=8)
    p.add_argument("--height", type=int, default=224)
    p.add_argument("--width", type=int, default=224)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--pin-memory", dest="pin_memory", action="store_true")
    p.add_argument("--prefetch", type=int, default=2)
    p.add_argument("--loader-steps", type=int, default=100)
    p.add_argument("--loader-samples", type=int, default=10000)
    # DDP
    # p.add_argument("--ddp", action="store_true", help="开启 NCCL all-reduce 基准（需 torchrun）")
    args = p.parse_args()

    if not torch.cuda.is_available():
        print("CUDA 不可用，退出。"); sys.exit(1)

    report = {"env": collect_env()}

    print("\n[1/5] 带宽测试…")
    report["bandwidth"] = test_bandwidth(args)
    for k,v in report["bandwidth"].items(): print(f"  {k}: {v:.2f}")

    print("\n[2/5] GEMM 吞吐…")
    report["gemm"] = test_gemm(args)
    def pprint_gemm(group):
        for n, res in report["gemm"][group].items():
            if "error" in res: print(f"  {group} n={n}: {res['error']}")
            else: print(f"  {group} n={n}: {res['TFLOPs']:.1f} TFLOP/s, {res['ms']:.3f} ms")
    for g in ["FP32_TF32_on", "FP32_TF32_off", "FP16", "BF16"]:
        pprint_gemm(g)

    print("\n[3/5] 数据加载与上卡吞吐…")
    report["input_pipeline"] = test_input_pipeline(args)
    for k,v in report["input_pipeline"].items(): print(f"  {k}: {v}")

    # # 仅在 torchrun + --ddp 时跑
    # if args.ddp:
    #     try:
    #         import torch.distributed as dist
    #         if dist.is_available() and dist.is_initialized():
    #             print("\n[4/5] NCCL All-Reduce…")
    #             comm = ddp_allreduce_bench(args)
    #             if dist.get_rank() == 0:
    #                 report["all_reduce"] = comm
    #                 for sz, m in comm.items():
    #                     print(f"  {sz}: {m}")
    #         else:
    #             print("\n[4/5] 跳过 NCCL（未初始化 torch.distributed）")
    #     except Exception as e:
    #         report["all_reduce_error"] = str(e)
    #         print(f"  NCCL 测试失败: {e}")
    # else:
    #     print("\n[4/5] 跳过 NCCL（未指定 --ddp）")

    # 保存 JSON
    fname = f"perf_report_{int(time.time())}.json"
    with open(fname, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n[5/5] 报告已保存: {fname}")

if __name__ == "__main__":
    main()
