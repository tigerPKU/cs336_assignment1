import sys
import os
import json
import numpy as np
from tqdm import tqdm

# 确保能导入 cs336_basics
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cs336_basics.tokenizer import Tokenizer


def pretokenize_file(tokenizer, input_file, output_file):
    print(f"Processing {input_file} -> {output_file}")

    # 统计行数用于进度条 (可选，文件极大时可能会慢，可以直接估算)
    # total_lines = sum(1 for _ in open(input_file, 'r', encoding='utf-8'))

    # 使用 chunk 读取以节省内存
    chunk_size = 1024 * 1024  # 每次读 1MB 文本

    # 临时列表 buffer
    buffer = []
    BUFFER_FLUSH_SIZE = 10_000_000  # 每 1000 万个 token 写入一次磁盘

    total_tokens = 0

    # 以追加模式打开二进制文件
    with open(output_file, "wb") as f_out:
        with open(input_file, "r", encoding="utf-8") as f_in:
            # 使用 Tokenizer 的 encode_iterable 可能是最优雅的，
            # 但为了控制写入磁盘的频率，我们这里手动分块处理

            # 简单的流式读取生成器
            def text_generator():
                while True:
                    chunk = f_in.read(chunk_size)
                    if not chunk:
                        break
                    yield chunk

            # 使用 tokenizer.encode_iterable 进行流式编码
            # 注意：这里假设你的 encode_iterable 已经通过测试并能正确处理流
            token_iterator = tokenizer.encode_iterable(text_generator())

            pbar = tqdm(desc="Tokenizing", unit=" tokens")

            for token_id in token_iterator:
                buffer.append(token_id)

                if len(buffer) >= BUFFER_FLUSH_SIZE:
                    # 转换为 uint16 并写入
                    arr = np.array(buffer, dtype=np.uint16)
                    f_out.write(arr.tobytes())
                    total_tokens += len(buffer)
                    pbar.update(len(buffer))
                    buffer = []

            # 写入剩余的 buffer
            if buffer:
                arr = np.array(buffer, dtype=np.uint16)
                f_out.write(arr.tobytes())
                total_tokens += len(buffer)
                pbar.update(len(buffer))

            pbar.close()

    print(f"Done. Total tokens: {total_tokens}")
    print(f"Saved to {output_file}")


def main():
    # 1. 加载 Tokenizer
    # 假设你之前训练的 tokenizer 保存在这里
    tokenizer_dir = "cs336_basics/tokenizers/tinystories_10k"
    vocab_path = os.path.join(tokenizer_dir, "vocab.json")
    merges_path = os.path.join(tokenizer_dir, "merges.json")

    if not os.path.exists(vocab_path):
        print(f"Error: Tokenizer not found at {tokenizer_dir}. Did you run scripts/train_tokenizer.py?")
        return

    tokenizer = Tokenizer.from_files(vocab_path, merges_path, special_tokens=["<|endoftext|>"])

    # 2. 定义数据路径
    data_dir = "data"
    tasks = [
        ("TinyStoriesV2-GPT4-train.txt", "tinystories_train.bin"),
        ("TinyStoriesV2-GPT4-valid.txt", "tinystories_valid.bin"),
    ]

    for input_name, output_name in tasks:
        input_path = os.path.join(data_dir, input_name)
        output_path = os.path.join(data_dir, output_name)

        if os.path.exists(input_path):
            pretokenize_file(tokenizer, input_path, output_path)
        else:
            print(f"Warning: Input file {input_path} not found.")


if __name__ == "__main__":
    main()
