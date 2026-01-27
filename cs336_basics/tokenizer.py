import regex as re
import os
import json
import multiprocessing
import gc
import shutil
from typing import List, Dict, Tuple, Iterable, Iterator
from collections import Counter, defaultdict
from tqdm import tqdm
from pathlib import Path

# GPT-2 Regex Pattern
GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def get_stats(vocab: Dict[Tuple[bytes, ...], int]) -> Dict[Tuple[bytes, bytes], int]:
    """统计相邻字节对频率"""
    pairs = defaultdict(int)
    for word, freq in vocab.items():
        for i in range(len(word) - 1):
            pair = (word[i], word[i + 1])
            pairs[pair] += freq
    return pairs


def _pretokenize_chunk(args):
    """Worker: 处理文本片段列表并返回词频"""
    fragments, pattern_str = args
    pat = re.compile(pattern_str)
    counts = Counter()

    # fragments 是一个 list[str]，每个 str 都是一个完整的连续文本块
    # 我们对每个块单独进行正则切分
    for text in fragments:
        for token in pat.findall(text):
            byte_tuple = tuple([bytes([b]) for b in token.encode("utf-8")])
            counts[byte_tuple] += 1
    return counts


def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: List[str],
    chunk_size_threshold: int = 10 * 1024 * 1024,  # 10MB accumulate buffer
    temp_dir: str = "temp_bpe_stats",
) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
    """
    修复版 BPE 训练：支持流式读取 + 正确处理跨行 Token (如 \\n\\n)
    """

    # === 1. 环境准备 ===
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)

    print(f"[Map Phase] Reading data from {input_path}...")

    special_token_set = set(special_tokens) if special_tokens else set()
    if special_tokens:
        sorted_special = sorted(special_tokens, key=len, reverse=True)
        # 使用捕获组 (...) 以便 split 后保留分隔符
        special_pattern_str = "(" + "|".join(re.escape(s) for s in sorted_special) + ")"
        special_re = re.compile(special_pattern_str)
    else:
        special_re = None

    # === 2. Map 阶段：智能分块读取 ===
    num_processes = max(1, multiprocessing.cpu_count() - 1)
    pool = multiprocessing.Pool(num_processes)

    # 用于累积即将发送给 Worker 的完整片段列表
    batch_fragments = []
    current_batch_size = 0

    # 用于累积当前正在读取的连续文本（尚未遇到 Special Token）
    # 使用 list[str] 比不断 string += string 更高效
    current_text_builder = []
    current_text_len = 0

    chunk_idx = 0

    def flush_batch():
        """将积累的 batch_fragments 发送给 Workers 并落盘"""
        nonlocal chunk_idx, batch_fragments, current_batch_size
        if not batch_fragments:
            return

        # 切分任务
        num_tasks = min(len(batch_fragments), num_processes * 4)
        task_size = len(batch_fragments) // num_tasks + 1
        chunks = [batch_fragments[i : i + task_size] for i in range(0, len(batch_fragments), task_size)]

        args = [(c, GPT2_SPLIT_PATTERN) for c in chunks]
        results = pool.map(_pretokenize_chunk, args)

        # 合并结果
        total_counts = Counter()
        for res in results:
            total_counts.update(res)

        # 序列化并落盘
        serializable_counts = {"".join(b.hex() for b in k): v for k, v in total_counts.items()}

        with open(os.path.join(temp_dir, f"part_{chunk_idx}.json"), "w") as tf:
            json.dump(serializable_counts, tf)

        chunk_idx += 1
        batch_fragments = []
        current_batch_size = 0
        gc.collect()

    with open(input_path, "r", encoding="utf-8") as f:
        pbar = tqdm(desc="Processing Stream")

        for line in f:
            pbar.update(len(line))

            # 如果没有特殊 Token，直接把行拼接到 builder
            if not special_re:
                current_text_builder.append(line)
                current_text_len += len(line)
            else:
                # 如果有特殊 Token，进行切分
                parts = special_re.split(line)
                # re.split 返回 [text, special, text, special, text...]

                # 第一部分：属于当前正在累积的 fragment
                if parts[0]:
                    current_text_builder.append(parts[0])
                    current_text_len += len(parts[0])

                # 如果 split 产生了多个部分，说明中间遇到了 Special Token
                if len(parts) > 1:
                    # 1. 完成当前的 fragment
                    full_frag = "".join(current_text_builder)
                    if full_frag:
                        batch_fragments.append(full_frag)
                        current_batch_size += len(full_frag)

                    # 重置 builder
                    current_text_builder = []
                    current_text_len = 0

                    # 2. 处理中间的部分
                    # parts[1], parts[3]... 是 Special Tokens (跳过)
                    # parts[2], parts[4]... 是 文本片段 (直接作为新 fragment)
                    for i in range(2, len(parts), 2):
                        frag = parts[i]
                        # 如果这不是最后一部分，它就是一个完整的被 Special Token 包围的片段
                        if i < len(parts) - 1:
                            if frag:
                                batch_fragments.append(frag)
                                current_batch_size += len(frag)
                        else:
                            # 最后一部分：是下一段文本的开始，放入 builder
                            if frag:
                                current_text_builder.append(frag)
                                current_text_len += len(frag)

            # 检查是否需要 Flush 到 Worker
            # 条件：Current Builder 太大（防止内存无限增长） 或 Batch 太大
            if current_text_len > chunk_size_threshold:
                # 强制切断当前文本（虽然可能切断单词，但 10MB 切一次概率很低，且是大文件训练所必需）
                full_frag = "".join(current_text_builder)
                batch_fragments.append(full_frag)
                current_batch_size += len(full_frag)
                current_text_builder = []
                current_text_len = 0

            if current_batch_size > chunk_size_threshold:
                flush_batch()

        # 循环结束，处理剩余数据
        if current_text_builder:
            batch_fragments.append("".join(current_text_builder))

        flush_batch()
        pbar.close()

    print(f"\n[Map Phase] Complete. Stats saved to {chunk_idx} files.")

    # === 3. Reduce 阶段：从硬盘加载并合并 ===
    print("[Reduce Phase] Merging stats from disk...")
    combined_counts = Counter()

    temp_files = list(Path(temp_dir).glob("*.json"))
    for tf in tqdm(temp_files, desc="Merging Files"):
        with open(tf, "r") as f:
            data = json.load(f)
            for k_hex, freq in data.items():
                raw_bytes = bytes.fromhex(k_hex)
                # 还原为 tuple
                key_tuple = tuple(bytes([b]) for b in raw_bytes)
                combined_counts[key_tuple] += freq

    shutil.rmtree(temp_dir)
    print(f"Number of unique pre-tokens: {len(combined_counts)}")

    # === 4. BPE 训练 (核心逻辑) ===
    vocab_list: List[List[int]] = []
    word_counts: List[int] = []

    vocab_id_map = {i: bytes([i]) for i in range(256)}
    next_id = 256
    for st in special_tokens:
        vocab_id_map[next_id] = st.encode("utf-8")
        next_id += 1

    for word_bytes_tuple, freq in combined_counts.items():
        ids = [int(b[0]) for b in word_bytes_tuple]
        vocab_list.append(ids)
        word_counts.append(freq)

    del combined_counts
    gc.collect()

    stats = defaultdict(int)
    indices = defaultdict(set)

    print("Building initial stats...")
    for i, word in enumerate(vocab_list):
        freq = word_counts[i]
        for j in range(len(word) - 1):
            pair = (word[j], word[j + 1])
            stats[pair] += freq
            indices[pair].add(i)

    merges = []
    num_merges = vocab_size - next_id
    print(f"Starting BPE loop. Need {num_merges} merges...")

    for i in tqdm(range(num_merges)):
        if not stats:
            break

        # Tie-breaking: 频率最高 > 字节序最大
        best_pair = max(stats, key=lambda p: (stats[p], vocab_id_map[p[0]], vocab_id_map[p[1]]))

        p0, p1 = best_pair
        merges.append((vocab_id_map[p0], vocab_id_map[p1]))

        new_token_bytes = vocab_id_map[p0] + vocab_id_map[p1]
        vocab_id_map[next_id] = new_token_bytes
        new_token_id = next_id
        next_id += 1

        words_to_update = indices[best_pair]

        for word_idx in list(words_to_update):
            word = vocab_list[word_idx]

            i = 0
            n = len(word)
            has_merge = False
            new_word = []

            while i < n:
                if i < n - 1 and word[i] == p0 and word[i + 1] == p1:
                    new_word.append(new_token_id)
                    has_merge = True
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1

            if not has_merge:
                continue

            freq = word_counts[word_idx]

            for j in range(len(word) - 1):
                pair = (word[j], word[j + 1])
                stats[pair] -= freq
                if stats[pair] == 0:
                    del stats[pair]

            for j in range(len(new_word) - 1):
                pair = (new_word[j], new_word[j + 1])
                stats[pair] += freq
                indices[pair].add(word_idx)

            vocab_list[word_idx] = new_word

        if best_pair in stats:
            del stats[best_pair]

    print(f"Training complete. Final vocab size: {len(vocab_id_map)}")
    return vocab_id_map, merges


# === Tokenizer 类 (保持缓存优化) ===
class Tokenizer:
    def __init__(self, vocab: Dict[int, bytes], merges: List[Tuple[bytes, bytes]], special_tokens: List[str] = None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens if special_tokens else []
        self.token_to_id = {v: k for k, v in vocab.items()}
        self.merges_rank = {pair: i for i, pair in enumerate(merges)}
        self.cache = {}
        self.special_token_ids = {}
        for st in self.special_tokens:
            st_bytes = st.encode("utf-8")
            if st_bytes in self.token_to_id:
                self.special_token_ids[st] = self.token_to_id[st_bytes]

        if self.special_tokens:
            self.special_pattern = (
                "(" + "|".join(re.escape(s) for s in sorted(self.special_tokens, key=len, reverse=True)) + ")"
            )
            self.special_re = re.compile(self.special_pattern)
        else:
            self.special_re = None

        self.gpt2_pat = re.compile(GPT2_SPLIT_PATTERN)

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: List[str] = None):
        with open(vocab_filepath, "r") as f:
            raw_vocab = json.load(f)
        vocab = {int(k): bytes.fromhex(v) for k, v in raw_vocab.items()}

        with open(merges_filepath, "r") as f:
            raw_merges = json.load(f)
        merges = [tuple(bytes.fromhex(b) for b in pair) for pair in raw_merges]

        return cls(vocab, merges, special_tokens)

    def _bpe(self, token_bytes: Tuple[bytes, ...]) -> Tuple[bytes, ...]:
        word = list(token_bytes)
        while len(word) >= 2:
            stats = get_stats({tuple(word): 1})
            pair_to_merge = None
            min_rank = float("inf")

            for pair in stats:
                if pair in self.merges_rank:
                    rank = self.merges_rank[pair]
                    if rank < min_rank:
                        min_rank = rank
                        pair_to_merge = pair

            if pair_to_merge is None:
                break

            p1, p2 = pair_to_merge
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == p1 and word[i + 1] == p2:
                    new_word.append(p1 + p2)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = new_word
        return tuple(word)

    def encode(self, text: str) -> List[int]:
        ids = []
        if self.special_re:
            parts = self.special_re.split(text)
        else:
            parts = [text]

        for part in parts:
            if not part:
                continue
            if part in self.special_token_ids:
                ids.append(self.special_token_ids[part])
                continue

            pre_tokens = self.gpt2_pat.findall(part)
            for token_str in pre_tokens:
                if token_str in self.cache:
                    ids.extend(self.cache[token_str])
                    continue

                token_bytes = tuple(bytes([b]) for b in token_str.encode("utf-8"))
                merged_bytes = self._bpe(token_bytes)

                current_ids = []
                for b in merged_bytes:
                    if b in self.token_to_id:
                        current_ids.append(self.token_to_id[b])

                self.cache[token_str] = current_ids
                ids.extend(current_ids)

        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for chunk in iterable:
            encoded_chunk = self.encode(chunk)
            for token_id in encoded_chunk:
                yield token_id

    def decode(self, ids: List[int]) -> str:
        byte_parts = []
        for i in ids:
            if i in self.vocab:
                byte_parts.append(self.vocab[i])
        combined_bytes = b"".join(byte_parts)
        return combined_bytes.decode("utf-8", errors="replace")
