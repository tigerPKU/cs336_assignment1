import os
import sys
import time
import math
import argparse
import numpy as np
import torch

# 将项目根目录加入路径，确保能导入 cs336_basics
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cs336_basics.transformer import TransformerLM, AdamW, cross_entropy, gradient_clipping, get_lr_cosine_schedule
from tests.adapters import run_get_batch, run_save_checkpoint


def get_args():
    parser = argparse.ArgumentParser(description="Train a Transformer LM")

    # 数据路径
    parser.add_argument("--train_path", type=str, required=True, help="Path to training .npy file")
    parser.add_argument("--val_path", type=str, required=True, help="Path to validation .npy file")
    parser.add_argument("--out_dir", type=str, default="checkpoints", help="Directory to save checkpoints")

    # 模型超参数 (默认值参考 TinyStories 配置)
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--d_ff", type=int, default=1344)
    parser.add_argument("--rope_theta", type=float, default=10000.0)

    # 训练超参数
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_iters", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--min_lr", type=float, default=6e-5)
    parser.add_argument("--warmup_iters", type=int, default=500)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)

    # 系统与日志
    parser.add_argument("--device", type=str, default=None, help="cuda, mps, or cpu")
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--eval_interval", type=int, default=200)
    parser.add_argument("--eval_iters", type=int, default=50)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--wandb_project", type=str, default=None, help="WandB project name")
    parser.add_argument("--wandb_run_name", type=str, default="transformer-run")

    return parser.parse_args()


@torch.no_grad()
def estimate_loss(model, data, batch_size, context_length, device, eval_iters):
    """
    估算验证集 Loss。
    """
    out = {}
    model.eval()
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        x, y = run_get_batch(data, batch_size, context_length, device)
        logits = model(x)
        loss = cross_entropy(logits, y)
        losses[k] = loss.item()
    model.train()
    return losses.mean().item()


def main():
    args = get_args()

    # 1. 设备配置
    if args.device:
        device = args.device
    else:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    print(f"Using device: {device}")

    # 2. 准备输出目录
    os.makedirs(args.out_dir, exist_ok=True)

    # 3. WandB 初始化
    if args.wandb_project:
        import wandb

        wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))

    # 4. 加载数据 (Memory Mapped)
    # 假设输入数据已经是 uint16 格式
    train_data = np.load(args.train_path, mmap_mode="r")
    val_data = np.load(args.val_path, mmap_mode="r")
    print(f"Loaded train data shape: {train_data.shape}")
    print(f"Loaded val data shape: {val_data.shape}")

    # 5. 初始化模型
    model = TransformerLM(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        max_seq_len=args.context_length,
        theta=args.rope_theta,
    )
    model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    # 6. 初始化优化器
    optimizer = AdamW(
        model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay, eps=1e-8
    )

    # 7. 训练循环
    print("Starting training...")
    t0 = time.time()

    for iter_num in range(1, args.max_iters + 1):
        # A. 设置学习率 (Cosine Schedule)
        lr = get_lr_cosine_schedule(iter_num, args.lr, args.min_lr, args.warmup_iters, args.max_iters)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # B. 获取 Batch
        x, y = run_get_batch(train_data, args.batch_size, args.context_length, device)

        # C. 前向传播与 Loss 计算
        logits = model(x)
        loss = cross_entropy(logits, y)

        # D. 反向传播与优化
        model.zero_grad(set_to_none=True)
        loss.backward()

        if args.grad_clip > 0.0:
            gradient_clipping(model.parameters(), args.grad_clip)

        optimizer.step()

        # E. 日志打印
        if iter_num % args.log_interval == 0:
            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            tokens_per_sec = (args.batch_size * args.context_length * args.log_interval) / dt
            print(
                f"iter {iter_num}: loss {loss.item():.4f}, time {dt * 1000:.2f}ms, lr {lr:.2e}, tok/s {tokens_per_sec:.0f}"
            )

            if args.wandb_project:
                wandb.log(
                    {
                        "train/loss": loss.item(),
                        "train/lr": lr,
                        "train/tokens_per_sec": tokens_per_sec,
                        "iter": iter_num,
                    }
                )

        # F. 验证集评估
        if iter_num % args.eval_interval == 0:
            val_loss = estimate_loss(model, val_data, args.batch_size, args.context_length, device, args.eval_iters)
            print(f"--- validation: iter {iter_num}, val loss {val_loss:.4f} ---")
            if args.wandb_project:
                wandb.log({"val/loss": val_loss, "iter": iter_num})

        # G. 保存 Checkpoint
        if iter_num % args.save_interval == 0 or iter_num == args.max_iters:
            ckpt_path = os.path.join(args.out_dir, f"ckpt_{iter_num}.pt")
            print(f"Saving checkpoint to {ckpt_path}")
            run_save_checkpoint(model, optimizer, iter_num, ckpt_path)

    print("Training finished.")


if __name__ == "__main__":
    main()
