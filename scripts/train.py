import os
import sys
import time
import math
import argparse
import numpy as np
import torch

# 将项目根目录加入路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cs336_basics.transformer import TransformerLM, AdamW, cross_entropy, gradient_clipping, get_lr_cosine_schedule
from tests.adapters import run_get_batch, run_save_checkpoint


def get_args():
    parser = argparse.ArgumentParser(description="Train a Transformer LM")

    # 数据路径
    parser.add_argument("--train_path", type=str, required=True, help="Path to training .npy file")
    parser.add_argument("--val_path", type=str, required=True, help="Path to validation .npy file")
    parser.add_argument("--out_dir", type=str, default="checkpoints", help="Directory to save checkpoints")

    # 模型超参数
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--d_ff", type=int, default=1344)
    parser.add_argument("--rope_theta", type=float, default=10000.0)

    # === Section 7 Ablation Flags (新增) ===
    parser.add_argument("--no_rms_norm", action="store_true", help="Ablation: Disable RMSNorm completely")
    parser.add_argument(
        "--norm_mode", type=str, default="pre", choices=["pre", "post"], help="Ablation: Pre-norm vs Post-norm"
    )
    parser.add_argument("--no_rope", action="store_true", help="Ablation: Disable RoPE (NoPE)")
    parser.add_argument(
        "--ffn_type", type=str, default="swiglu", choices=["swiglu", "silu"], help="Ablation: SwiGLU vs SiLU"
    )

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
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--eval_interval", type=int, default=200)
    parser.add_argument("--eval_iters", type=int, default=50)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--wandb_project", type=str, default=None, help="WandB project name")
    parser.add_argument("--wandb_run_name", type=str, default=None)

    return parser.parse_args()


@torch.no_grad()
def estimate_loss(model, data, batch_size, context_length, device, eval_iters):
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

    # 构造 WandB Run Name
    run_name = args.wandb_run_name
    if run_name is None:
        if args.no_rms_norm:
            run_name = "ablation_no_rmsnorm"
        elif args.norm_mode == "post":
            run_name = "ablation_post_norm"
        elif args.no_rope:
            run_name = "ablation_no_rope"
        elif args.ffn_type == "silu":
            run_name = "ablation_silu"
        else:
            run_name = "baseline"

    # 初始化 WandB
    if args.wandb_project:
        import wandb

        wandb.init(project=args.wandb_project, name=run_name, config=vars(args))

    os.makedirs(args.out_dir, exist_ok=True)

    # 加载数据
    train_data = np.load(args.train_path, mmap_mode="r")
    val_data = np.load(args.val_path, mmap_mode="r")

    # 初始化模型
    # 将新参数传入 TransformerLM
    model = TransformerLM(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        max_seq_len=args.context_length,
        theta=args.rope_theta,
        use_rms_norm=not args.no_rms_norm,  # 注意这里取反
        norm_mode=args.norm_mode,
        ffn_type=args.ffn_type,
        use_rope=not args.no_rope,  # 注意这里取反
    )
    model.to(device)
    print(f"Model params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    optimizer = AdamW(
        model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), weight_decay=args.weight_decay, eps=1e-8
    )

    print(f"Starting training: {run_name}")
    t0 = time.time()

    for iter_num in range(1, args.max_iters + 1):
        lr = get_lr_cosine_schedule(iter_num, args.lr, args.min_lr, args.warmup_iters, args.max_iters)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        x, y = run_get_batch(train_data, args.batch_size, args.context_length, device)
        logits = model(x)
        loss = cross_entropy(logits, y)

        model.zero_grad(set_to_none=True)
        loss.backward()

        if args.grad_clip > 0.0:
            gradient_clipping(model.parameters(), args.grad_clip)

        optimizer.step()

        if iter_num % args.log_interval == 0:
            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            tokens_per_sec = (args.batch_size * args.context_length * args.log_interval) / dt
            print(f"Iter {iter_num}: Loss {loss.item():.4f}, LR {lr:.2e}, Tok/s {tokens_per_sec:.0f}")
            if args.wandb_project:
                wandb.log(
                    {"train/loss": loss.item(), "train/lr": lr, "train/tokens_per_sec": tokens_per_sec}, step=iter_num
                )

        if iter_num % args.eval_interval == 0:
            val_loss = estimate_loss(model, val_data, args.batch_size, args.context_length, device, args.eval_iters)
            print(f"--- Eval {iter_num}: Val Loss {val_loss:.4f} ---")
            if args.wandb_project:
                wandb.log({"val/loss": val_loss}, step=iter_num)

        if iter_num % args.save_interval == 0 or iter_num == args.max_iters:
            ckpt_path = os.path.join(args.out_dir, f"ckpt_{run_name}_{iter_num}.pt")
            run_save_checkpoint(model, optimizer, iter_num, ckpt_path)

    print("Training finished.")
    if args.wandb_project:
        wandb.finish()


if __name__ == "__main__":
    main()
