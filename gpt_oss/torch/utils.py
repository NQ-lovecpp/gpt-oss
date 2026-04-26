import os
import torch
import torch.distributed as dist


def suppress_output(rank):
    """Suppress printing on the current device. Force printing with `force=True`."""
    # 通过 monkey patch 内置 print，实现多卡场景下“仅 rank0 默认打印”。
    # 调试时可传 force=True 强制打印当前 rank 的日志。
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if force:
            builtin_print("rank #%d:" % rank, *args, **kwargs)
        elif rank == 0:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def init_distributed() -> torch.device:
    """Initialize the model for distributed inference."""
    # Initialize distributed inference
    # 约定从环境变量读取分布式配置；单卡时 WORLD_SIZE 默认为 1。
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    if world_size > 1:
        # 使用 NCCL + env:// 初始化进程组（常见于 torchrun 启动）。
        dist.init_process_group(
            backend="nccl", init_method="env://", world_size=world_size, rank=rank
        )
    # 将当前进程绑定到对应 GPU，避免跨卡误用。
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    # Warm up NCCL to avoid first-time latency
    if world_size > 1:
        # 预热一次 all_reduce，降低首次通信抖动。
        x = torch.ones(1, device=device)
        dist.all_reduce(x)
        torch.cuda.synchronize(device)

    # 最后统一设置打印策略，减少重复日志噪声。
    suppress_output(rank)
    return device
