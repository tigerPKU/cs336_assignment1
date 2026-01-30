import sys
import os
import numpy as np

# 将项目根目录加入 python path，确保能找到 cs336_basics
# 假设脚本在 scripts/ 目录下，.. 就是项目根目录
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cs336_basics.tokenizer import Tokenizer

# ================= 配置区域 =================
# 1. 验证集原始文本路径 (请修改为你服务器上的实际路径)
TINYSTORIES_VAL_INPUT = "/home1/jym/sftp/LLM_2026/A1/data/TinyStoriesV2-GPT4-valid.txt"
OWT_VAL_INPUT = "/home1/jym/sftp/LLM_2026/A1/data/owt_valid.txt"  # 如果没有可以忽略

# 2. 之前生成的 Tokenizer 路径 (不需要修改，除非你改了 prepare_data.py 的输出)
DATA_DIR = "data_preprocessed"
SPECIAL_TOKENS = ["<|endoftext|>"]
# ===========================================


def save_dataset_as_npy(tokenizer, input_path, output_path):
    """
    读取文本文件，使用 tokenizer 编码，并保存为 numpy uint16 格式
    """
    print(f"正在处理: {input_path} -> {output_path}")
    ids = []

    # 检查输入文件是否存在
    if not os.path.exists(input_path):
        print(f"警告: 找不到输入文件 {input_path}，跳过。")
        return

    with open(input_path, "r", encoding="utf-8") as f:
        # 为了显示进度，尝试导入 tqdm，如果没有则使用普通迭代
        try:
            from tqdm import tqdm

            iterator = tqdm(f, desc=f"Encoding {os.path.basename(input_path)}")
        except ImportError:
            iterator = f
            print("提示: 安装 tqdm 可以看到进度条 (pip install tqdm)")

        for line in iterator:
            if not line.strip():
                continue
            # 编码每一行
            line_ids = tokenizer.encode(line)
            ids.extend(line_ids)
            # 添加 EOS token
            ids.append(tokenizer.special_token_ids["<|endoftext|>"])

    # 转换为 uint16 (节省内存，与训练集格式保持一致)
    ids_arr = np.array(ids, dtype=np.uint16)

    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    np.save(output_path, ids_arr)
    print(f"完成! 数据已保存: {output_path}")
    print(f"  Shape: {ids_arr.shape}")
    print(f"  Dtype: {ids_arr.dtype}")


def main():
    print("=== 开始处理验证集 (Validation Set) ===")

    # ---------------------------------------------------------
    # 任务 1: TinyStories 验证集
    # ---------------------------------------------------------
    vocab_path = os.path.join(DATA_DIR, "tinystories_vocab.json")
    merges_path = os.path.join(DATA_DIR, "tinystories_merges.json")
    val_output_path = os.path.join(DATA_DIR, "tinystories_val.npy")

    if os.path.exists(vocab_path) and os.path.exists(merges_path):
        print(f"\n[TinyStories] 加载 Tokenizer: {vocab_path}")
        # 关键点：使用 from_files 加载现有的 tokenizer
        ts_tok = Tokenizer.from_files(vocab_path, merges_path, SPECIAL_TOKENS)

        save_dataset_as_npy(ts_tok, TINYSTORIES_VAL_INPUT, val_output_path)
    else:
        print(f"\n[TinyStories] 错误: 找不到 Tokenizer 文件。请先运行 prepare_data.py！")

    # ---------------------------------------------------------
    # 任务 2: OpenWebText 验证集 (可选)
    # ---------------------------------------------------------
    owt_vocab = os.path.join(DATA_DIR, "owt_vocab.json")
    owt_merges = os.path.join(DATA_DIR, "owt_merges.json")
    owt_val_output = os.path.join(DATA_DIR, "owt_val.npy")

    # 只有当 OWT Tokenizer 和 验证集文件都存在时才运行
    if os.path.exists(owt_vocab) and os.path.exists(owt_merges) and os.path.exists(OWT_VAL_INPUT):
        print(f"\n[OpenWebText] 加载 Tokenizer: {owt_vocab}")
        owt_tok = Tokenizer.from_files(owt_vocab, owt_merges, SPECIAL_TOKENS)
        save_dataset_as_npy(owt_tok, OWT_VAL_INPUT, owt_val_output)
    else:
        print("\n[OpenWebText] 跳过 (未找到 Tokenizer 或 输入文件)")


if __name__ == "__main__":
    main()
