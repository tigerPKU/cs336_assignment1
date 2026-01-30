"""
脚本名称: prepare_data.py
功能: 完成 Assignment 1 Section 2 的所有 Tokenizer 训练、实验和数据预处理任务。
"""

import sys
import os
import time
import numpy as np

# 将项目根目录加入 python path，确保能找到 cs336_basics
# 假设脚本在 scripts/ 目录下，.. 就是根目录
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cs336_basics.tokenizer import train_bpe, Tokenizer

# ================= 配置区域 =================
# 请修改为你的实际文件路径
# 如果没有完整数据，可以用小的 sample 文件代替跑通流程
# ================= 配置区域 =================
# 修改为你的实际绝对路径
TINYSTORIES_PATH = "/home1/jym/sftp/LLM_2026/A1/data/TinyStoriesV2-GPT4-train.txt"
OWT_PATH = "/home1/jym/sftp/LLM_2026/A1/data/owt_train.txt"

# 输出目录 (保持默认即可，生成的 vocab 和 .npy 会存在你当前项目的 data_preprocessed 文件夹下)
OUTPUT_DIR = "data_preprocessed"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SPECIAL_TOKENS = ["<|endoftext|>"]


def run_tokenizer_experiment(dataset_name: str, input_path: str, vocab_size: int, sample_text_for_ratio: str):
    print(f"\n=== 开始任务: 训练 {dataset_name} Tokenizer (Vocab={vocab_size}) ===")

    # 1. 训练 BPE
    t0 = time.time()
    vocab, merges = train_bpe(input_path, vocab_size, SPECIAL_TOKENS)
    t1 = time.time()
    print(f"[{dataset_name}] 训练完成，耗时: {t1 - t0:.2f}秒")

    # 2. 保存 Tokenizer
    vocab_path = os.path.join(OUTPUT_DIR, f"{dataset_name}_vocab.json")
    merges_path = os.path.join(OUTPUT_DIR, f"{dataset_name}_merges.json")

    # 这里我们简单保存为 json 格式以便 Tokenizer.from_files 读取
    # 注意：你需要确保你的 tokenizer.py 中有保存 json 的辅助函数，
    # 或者我们在这里手动保存
    import json

    # key (int) -> hex string (bytes)
    vocab_export = {k: v.hex() for k, v in vocab.items()}
    # (bytes, bytes) -> (hex, hex)
    merges_export = [(p[0].hex(), p[1].hex()) for p in merges]

    with open(vocab_path, "w") as f:
        json.dump(vocab_export, f)
    with open(merges_path, "w") as f:
        json.dump(merges_export, f)
    print(f"[{dataset_name}] 模型已保存到 {OUTPUT_DIR}")

    # 3. 重新加载 Tokenizer 进行测试
    tok = Tokenizer.from_files(vocab_path, merges_path, SPECIAL_TOKENS)

    # 4. 分析最长 Token
    longest_token_id = max(tok.vocab.keys(), key=lambda k: len(tok.vocab[k]))
    longest_token_bytes = tok.vocab[longest_token_id]
    print(
        f"[{dataset_name}] 最长 Token (ID {longest_token_id}): {longest_token_bytes} (长度 {len(longest_token_bytes)})"
    )
    try:
        print(f"   -> 解码文本: {longest_token_bytes.decode('utf-8', errors='replace')}")
    except:
        pass

    # 5. 计算压缩率 (Compression Ratio)
    # Ratio = 原始 UTF-8 字节数 / 编码后的 Token 数量
    raw_bytes = len(sample_text_for_ratio.encode("utf-8"))
    encoded_ids = tok.encode(sample_text_for_ratio)
    num_tokens = len(encoded_ids)
    ratio = raw_bytes / num_tokens if num_tokens > 0 else 0
    print(f"[{dataset_name}] 压缩率: {ratio:.4f} bytes/token")

    return tok


def save_dataset_as_npy(tokenizer, input_path, output_filename):
    print(f"正在将 {input_path} 转换为 numpy 格式...")
    ids = []
    # 逐行读取以节省内存
    with open(input_path, "r", encoding="utf-8") as f:
        # 正式跑：使用 tqdm 显示进度，并且不要 break
        from tqdm import tqdm

        for line in tqdm(f, desc=f"Processing {output_filename}"):
            if not line.strip():
                continue
            line_ids = tokenizer.encode(line)
            ids.extend(line_ids)
            ids.append(tokenizer.special_token_ids["<|endoftext|>"])

            # # 为了演示，只处理少量数据防止卡死，实际使用请删除下面两行
            # if i > 5000:
            #     break

    # 转换为 uint16 (节省内存，最大支持 65535 vocab size)
    ids_arr = np.array(ids, dtype=np.uint16)
    save_path = os.path.join(OUTPUT_DIR, output_filename)
    np.save(save_path, ids_arr)
    print(f"数据已保存: {save_path}, Shape: {ids_arr.shape}, Dtype: {ids_arr.dtype}")


def main():
    # 读取采样文本用于计算压缩率 (简单读取文件前 10k 字符)
    with open(TINYSTORIES_PATH, "r", encoding="utf-8") as f:
        ts_sample = f.read(10000)

    # 1. 运行 TinyStories 实验 (Vocab 10k)
    ts_tok = run_tokenizer_experiment("tinystories", TINYSTORIES_PATH, 10000, ts_sample)

    # 2. 预处理 TinyStories 数据 (保存为 train.npy)
    save_dataset_as_npy(ts_tok, TINYSTORIES_PATH, "tinystories_train.npy")

    # 如果有 OWT 数据，取消下面注释运行
    if os.path.exists(OWT_PATH):
        with open(OWT_PATH, "r", encoding="utf-8") as f:
            owt_sample = f.read(10000)

        # 3. 运行 OpenWebText 实验 (Vocab 32k)
        owt_tok = run_tokenizer_experiment("owt", OWT_PATH, 32000, owt_sample)

        # 4. 交叉实验: 用 TinyStories Tokenizer 编码 OWT 数据
        print("\n=== 交叉实验: TinyStories Tokenizer on OWT ===")
        raw_bytes = len(owt_sample.encode("utf-8"))
        encoded_ids = ts_tok.encode(owt_sample)
        ratio = raw_bytes / len(encoded_ids)
        print(f"压缩率 (TS Tokenizer on OWT): {ratio:.4f} bytes/token (通常会比 OWT Tokenizer 差)")

        # 5. 预处理 OWT 数据
        save_dataset_as_npy(owt_tok, OWT_PATH, "owt_train.npy")


if __name__ == "__main__":
    main()
