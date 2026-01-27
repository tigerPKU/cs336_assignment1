import os
import sys
import torch
import argparse
import time

# 将项目根目录加入路径，确保能找到 cs336_basics
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cs336_basics.transformer import TransformerLM
from cs336_basics.tokenizer import Tokenizer


def get_args():
    parser = argparse.ArgumentParser(description="Generate text from trained model")

    # === 路径参数 ===
    # 必须提供模型 checkpoint 路径
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to model checkpoint (.pt)")

    # 默认指向 prepare_data.py 生成的文件路径
    parser.add_argument(
        "--vocab_path", type=str, default="data_preprocessed/tinystories_vocab.json", help="Path to vocab.json"
    )
    parser.add_argument(
        "--merges_path", type=str, default="data_preprocessed/tinystories_merges.json", help="Path to merges.json"
    )

    # === 生成参数 ===
    parser.add_argument("--prompt", type=str, default="Once upon a time", help="Starting prompt")
    parser.add_argument("--max_new_tokens", type=int, default=200, help="Maximum number of tokens to generate")
    parser.add_argument(
        "--temperature", type=float, default=0.8, help="Sampling temperature. 0.0 means greedy decoding"
    )
    parser.add_argument("--top_p", type=float, default=0.9, help="Nucleus sampling threshold")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed")

    # === 模型结构参数 ===
    # 注意：这里的默认值必须与你 train.py 中的设置完全一致
    # 如果你修改了 train.py 的参数，这里也要对应修改，或者在命令行传入
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--d_ff", type=int, default=1344)
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--rope_theta", type=float, default=10000.0)

    # === 系统参数 ===
    parser.add_argument("--device", type=str, default=None, help="cuda, mps, or cpu")

    return parser.parse_args()


def main():
    args = get_args()

    # 1. 设置随机种子和设备
    torch.manual_seed(args.seed)
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

    # 2. 加载 Tokenizer
    # prepare_data.py 保存的是 json 格式，直接使用 from_files 加载
    print(f"Loading tokenizer from {args.vocab_path} and {args.merges_path}...")
    if not os.path.exists(args.vocab_path) or not os.path.exists(args.merges_path):
        raise FileNotFoundError("Tokenizer files not found. Did you run prepare_data.py?")

    tokenizer = Tokenizer.from_files(
        vocab_filepath=args.vocab_path, merges_filepath=args.merges_path, special_tokens=["<|endoftext|>"]
    )
    print(f"Tokenizer loaded. Vocab size: {len(tokenizer.vocab)}")

    # 3. 初始化模型结构
    print("Initializing model...")
    model = TransformerLM(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        max_seq_len=args.context_length,
        theta=args.rope_theta,
    )

    # 4. 加载 Checkpoint 权重
    print(f"Loading checkpoint from {args.checkpoint_path}...")
    if not os.path.exists(args.checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {args.checkpoint_path}")

    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    state_dict = checkpoint["model_state_dict"]

    # 处理可能的 _orig_mod 前缀 (如果训练时使用了 torch.compile)
    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)

    # 加载权重 (strict=True 确保结构完全匹配)
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as e:
        print(f"Warning: strict loading failed ({e}), trying strict=False...")
        model.load_state_dict(state_dict, strict=False)

    model.to(device)
    model.eval()

    # 5. 准备 Prompt
    print(f"\nPrompt: '{args.prompt}'")
    prompt_ids = tokenizer.encode(args.prompt)
    x = torch.tensor(prompt_ids, dtype=torch.long, device=device)[None, :]  # Shape: (1, seq_len)

    # 获取 EOS Token ID 用于提前停止
    eos_id = tokenizer.special_token_ids.get("<|endoftext|>", None)

    # 6. 生成文本
    print("Generating...")
    t0 = time.time()

    # 调用模型内部的 generate 函数
    # 注意：确保你在 TransformerLM 类中已经实现了 generate 方法
    y = model.generate(
        x, max_new_tokens=args.max_new_tokens, temperature=args.temperature, top_p=args.top_p, eos_id=eos_id
    )
    t1 = time.time()

    # 7. 解码并输出
    output_ids = y[0].tolist()
    generated_text = tokenizer.decode(output_ids)

    print(f"\n{'=' * 20} Generated Text {'=' * 20}")
    print(generated_text)
    print(f"{'=' * 56}")
    print(f"Time: {t1 - t0:.2f}s | Tokens generated: {len(output_ids) - len(prompt_ids)}")


if __name__ == "__main__":
    main()
