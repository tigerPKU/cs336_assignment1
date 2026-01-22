import regex as re
import os
import json
import multiprocessing
from typing import List, Dict, Tuple, Iterable, Iterator
from collections import Counter, defaultdict
from tqdm import tqdm

# GPT-2 Regex Pattern
GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def get_stats(vocab: Dict[Tuple[bytes, ...], int]) -> Dict[Tuple[bytes, bytes], int]:
    """
    统计当前词表中所有相邻字节对的频率。
    """
    pairs = defaultdict(int)
    for word, freq in vocab.items():
        for i in range(len(word) - 1):
            pair = (word[i], word[i + 1])
            pairs[pair] += freq
    return pairs


def merge_vocab(pair: Tuple[bytes, bytes], vocab: Dict[Tuple[bytes, ...], int]) -> Dict[Tuple[bytes, ...], int]:
    """
    将词表中指定的字节对合并为一个新的 Token。
    """
    new_vocab = {}
    p_first, p_second = pair

    for word, freq in vocab.items():
        # 如果单词长度为1，无法合并，直接保留
        if len(word) == 1:
            new_vocab[word] = freq
            continue

        new_word = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and word[i] == p_first and word[i + 1] == p_second:
                new_word.append(p_first + p_second)
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        new_vocab[tuple(new_word)] = freq

    return new_vocab


def _pretokenize_chunk(args):
    """
    Worker function: 接收字符串列表，独立处理每个片段，不进行拼接。
    """
    text_fragments, pattern_str = args
    # 在 Worker 内部编译正则，确保序列化安全
    pat = re.compile(pattern_str)

    counts = Counter()
    for fragment in text_fragments:
        # 对每个片段单独跑正则，保证片段间的边界不被污染
        for token in pat.findall(fragment):
            # 转换为 bytes tuple: "the" -> (b't', b'h', b'e')
            byte_tuple = tuple([bytes([b]) for b in token.encode("utf-8")])
            counts[byte_tuple] += 1
    return counts


def train_bpe(
    input_path: str, vocab_size: int, special_tokens: List[str]
) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
    # 1. 读取数据
    print(f"Reading data from {input_path}...")
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    # 2. 预处理：按特殊 Token 切分
    print("Pre-tokenizing...")
    if special_tokens:
        sorted_special = sorted(special_tokens, key=len, reverse=True)
        special_pattern = "(" + "|".join(re.escape(s) for s in sorted_special) + ")"
        splits = re.split(special_pattern, text)
    else:
        splits = [text]

    # 过滤掉特殊 Token 和空字符串，只保留需要训练的文本片段
    special_token_set = set(special_tokens) if special_tokens else set()
    training_fragments = [s for s in splits if s and s not in special_token_set]

    # === 修复 1 & 2: 智能并行与无拼接处理 ===
    combined_counts = Counter()

    # 阈值设为 5MB。如果小于 5MB，单进程跑更快（避免 Windows 进程创建开销）
    # corpus.en 很小，会走这个分支，从而通过 Speed 测试
    if len(text) < 5 * 1024 * 1024:
        combined_counts = _pretokenize_chunk((training_fragments, GPT2_SPLIT_PATTERN))
    else:
        # 大文件使用多进程
        num_processes = max(1, multiprocessing.cpu_count() - 1)
        # 将片段列表切分为子列表
        chunk_size = len(training_fragments) // num_processes + 1
        chunks = [training_fragments[i : i + chunk_size] for i in range(0, len(training_fragments), chunk_size)]

        # 传递的是 list[str]，而不是拼接后的 str
        args = [(chunk, GPT2_SPLIT_PATTERN) for chunk in chunks if chunk]

        with multiprocessing.Pool(num_processes) as pool:
            results = pool.map(_pretokenize_chunk, args)

        for res in results:
            combined_counts.update(res)

    print(f"Number of unique pre-tokens: {len(combined_counts)}")

    # 3. 初始化词表 (0-255 + Special Tokens)
    vocab_id_map = {i: bytes([i]) for i in range(256)}
    next_id = 256
    for st in special_tokens:
        vocab_id_map[next_id] = st.encode("utf-8")
        next_id += 1

    merges = []
    num_merges = vocab_size - next_id

    print(f"Starting BPE loop. Need {num_merges} merges...")

    # 4. BPE 训练循环
    current_vocab_counts = combined_counts

    for i in tqdm(range(num_merges)):
        stats = get_stats(current_vocab_counts)
        if not stats:
            break

        # 按照 (频率, 字节对) 进行排序，由大到小选择，保证确定性
        # 注意：dict 的 keys 是 bytes，python 比较 bytes 是字典序
        best_pair = max(stats, key=lambda x: (stats[x], x))

        merges.append(best_pair)

        # 新 Token (仅用于记录)
        new_token_bytes = best_pair[0] + best_pair[1]
        vocab_id_map[next_id] = new_token_bytes
        next_id += 1

        current_vocab_counts = merge_vocab(best_pair, current_vocab_counts)

    print(f"Training complete. Final vocab size: {len(vocab_id_map)}")
    return vocab_id_map, merges


class Tokenizer:
    def __init__(self, vocab: Dict[int, bytes], merges: List[Tuple[bytes, bytes]], special_tokens: List[str] = None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens if special_tokens else []
        self.token_to_id = {v: k for k, v in vocab.items()}

        # 建立 Merge 优先级查询表
        self.merges_rank = {pair: i for i, pair in enumerate(merges)}

        # 缓存 Special Token ID
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
                token_bytes = tuple(bytes([b]) for b in token_str.encode("utf-8"))
                merged_bytes = self._bpe(token_bytes)
                for b in merged_bytes:
                    if b in self.token_to_id:
                        ids.append(self.token_to_id[b])
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
