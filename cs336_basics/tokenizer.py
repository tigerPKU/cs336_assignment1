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
    主要用于 Tokenizer 在推理阶段（_bpe）对单个单词进行统计。
    """
    pairs = defaultdict(int)
    for word, freq in vocab.items():
        for i in range(len(word) - 1):
            pair = (word[i], word[i + 1])
            pairs[pair] += freq
    return pairs


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
    """
    使用优化的增量更新算法训练 BPE Tokenizer。
    """
    # 1. 读取数据
    print(f"Reading data from {input_path}...")
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    # 2. 预处理：处理特殊 Token 并切分
    print("Pre-tokenizing...")
    special_token_set = set(special_tokens) if special_tokens else set()
    if special_tokens:
        sorted_special = sorted(special_tokens, key=len, reverse=True)
        special_pattern = "(" + "|".join(re.escape(s) for s in sorted_special) + ")"
        splits = re.split(special_pattern, text)
    else:
        splits = [text]

    training_fragments = [s for s in splits if s and s not in special_token_set]

    # 多进程统计预分词后的词频
    combined_counts = Counter()
    if len(text) < 5 * 1024 * 1024:
        # 小文件直接处理
        combined_counts = _pretokenize_chunk((training_fragments, GPT2_SPLIT_PATTERN))
    else:
        # 大文件多进程处理
        num_processes = max(1, multiprocessing.cpu_count() - 1)
        chunk_size = len(training_fragments) // num_processes + 1
        chunks = [training_fragments[i : i + chunk_size] for i in range(0, len(training_fragments), chunk_size)]
        args = [(chunk, GPT2_SPLIT_PATTERN) for chunk in chunks if chunk]
        with multiprocessing.Pool(num_processes) as pool:
            results = pool.map(_pretokenize_chunk, args)
        for res in results:
            combined_counts.update(res)

    print(f"Number of unique pre-tokens: {len(combined_counts)}")

    # === 初始化 BPE 数据结构 ===
    # vocab_list: 存储单词的 Token ID 列表，例如 [[id1, id2], [id3, id4]]
    vocab_list: List[List[int]] = []
    # word_counts: 对应单词的频率
    word_counts: List[int] = []

    # 初始化基础词表 (0-255 + Special Tokens)
    vocab_id_map = {i: bytes([i]) for i in range(256)}
    next_id = 256
    for st in special_tokens:
        vocab_id_map[next_id] = st.encode("utf-8")
        next_id += 1

    # 将统计结果填入列表
    for word_bytes_tuple, freq in combined_counts.items():
        ids = [int(b[0]) for b in word_bytes_tuple]
        vocab_list.append(ids)
        word_counts.append(freq)

    # stats: 记录相邻字节对的频率 (id1, id2) -> freq
    stats = defaultdict(int)
    # indices: 倒排索引，记录字节对出现在哪些单词中 (id1, id2) -> {word_idx, ...}
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

    # === BPE 训练主循环 ===
    for i in tqdm(range(num_merges)):
        if not stats:
            break

        # 选择最佳合并对：
        # 1. 频率最高
        # 2. 频率相同时，选择字节序（lexicographically）最大的
        best_pair = max(stats, key=lambda p: (stats[p], vocab_id_map[p[0]], vocab_id_map[p[1]]))

        # 记录 Merge 操作
        p0, p1 = best_pair
        merges.append((vocab_id_map[p0], vocab_id_map[p1]))

        # 生成新 Token
        new_token_bytes = vocab_id_map[p0] + vocab_id_map[p1]
        vocab_id_map[next_id] = new_token_bytes
        new_token_id = next_id
        next_id += 1

        # === 增量更新 ===
        # 获取所有可能包含 best_pair 的单词索引 (indices 包含的是超集)
        words_to_update = indices[best_pair]

        for word_idx in list(words_to_update):
            word = vocab_list[word_idx]

            # 扫描单词，检查是否存在 best_pair 并执行合并
            i = 0
            n = len(word)
            has_merge = False
            new_word = []

            # 构建新单词
            while i < n:
                if i < n - 1 and word[i] == p0 and word[i + 1] == p1:
                    new_word.append(new_token_id)
                    has_merge = True
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1

            # 如果该单词中并没有实际发生合并（可能是 indices 中的过期索引），直接跳过
            if not has_merge:
                continue

            # === 更新统计信息 ===
            freq = word_counts[word_idx]

            # 1. 扣除旧邻居的频率 (Old Pairs)
            for j in range(len(word) - 1):
                pair = (word[j], word[j + 1])
                stats[pair] -= freq
                if stats[pair] == 0:
                    del stats[pair]

            # 2. 加上新邻居的频率 (New Pairs)
            for j in range(len(new_word) - 1):
                pair = (new_word[j], new_word[j + 1])
                stats[pair] += freq
                indices[pair].add(word_idx)

            # 更新词表中的单词
            vocab_list[word_idx] = new_word

        # 清理 best_pair
        if best_pair in stats:
            del stats[best_pair]

    print(f"Training complete. Final vocab size: {len(vocab_id_map)}")
    return vocab_id_map, merges


class Tokenizer:
    def __init__(self, vocab: Dict[int, bytes], merges: List[Tuple[bytes, bytes]], special_tokens: List[str] = None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens if special_tokens else []
        self.token_to_id = {v: k for k, v in vocab.items()}

        # 建立 Merge 优先级查询表，用于推理时快速查找最小 Rank
        self.merges_rank = {pair: i for i, pair in enumerate(merges)}

        # === 修复点：初始化缓存字典 ===
        self.cache = {}

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
        # JSON key 必须是 string，这里转回 int
        vocab = {int(k): bytes.fromhex(v) for k, v in raw_vocab.items()}

        with open(merges_filepath, "r") as f:
            raw_merges = json.load(f)
        merges = [tuple(bytes.fromhex(b) for b in pair) for pair in raw_merges]

        return cls(vocab, merges, special_tokens)

    def _bpe(self, token_bytes: Tuple[bytes, ...]) -> Tuple[bytes, ...]:
        """
        对单个预分词 Token 应用 BPE 规则进行合并。
        """
        word = list(token_bytes)
        while len(word) >= 2:
            # 统计当前单词内的所有对子
            stats = get_stats({tuple(word): 1})
            pair_to_merge = None
            min_rank = float("inf")

            # 找到 Rank 最小（最早被学习到）的对子
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
        # 1. 处理 Special Tokens 切分
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

            # 2. GPT-2 预分词
            pre_tokens = self.gpt2_pat.findall(part)
            for token_str in pre_tokens:
                # === 优化：查缓存 ===
                if token_str in self.cache:
                    ids.extend(self.cache[token_str])
                    continue

                # 3. 转换为字节并应用 BPE
                token_bytes = tuple(bytes([b]) for b in token_str.encode("utf-8"))
                merged_bytes = self._bpe(token_bytes)

                current_ids = []
                for b in merged_bytes:
                    if b in self.token_to_id:
                        current_ids.append(self.token_to_id[b])
                    else:
                        pass  # 理论上不应发生

                # === 优化：写入缓存 ===
                self.cache[token_str] = current_ids
                ids.extend(current_ids)

        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        流式编码，适用于大数据集
        """
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
        # errors="replace" 会将无法解码的字节替换为 U+FFFD
        return combined_bytes.decode("utf-8", errors="replace")
