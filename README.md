

# CS336: Language Modeling from Scratch

This repository contains my implementation of **Stanford's CS336: Language Modeling from Scratch (Spring 2025)**. This project focuses on building, training, and optimizing large language models from the ground up.

> [!IMPORTANT]
> **Academic Integrity:** This repository is for educational and portfolio purposes only. If you are a current student of CS336, please adhere to your institution's honor code regarding public code sharing.

---

## 📝 Assignment 1: Basics

In the first assignment, I implemented a Transformer-based language model and its corresponding tokenizer from scratch.

### Core Implementation

* **Byte Pair Encoding (BPE):** Custom implementation for tokenizer training, encoding, and decoding.
* **Transformer Architecture:** Features Multi-head Self-Attention, RMSNorm, SwiGLU activation, and Rotary Positional Embeddings (RoPE).
* **Training Pipeline:** Comprehensive training loops with AdamW optimizer, gradient clipping, and learning rate scheduling.

### 🚀 Experiment Writeup

For detailed analysis, hyperparameter sweeps (LR Sweep), and training performance on `TinyStories`, please refer to:
👉 **[Assignment 1 Writeup](./writeup/writeup.md)**

---

## 📂 Repository Structure

```text
.
├── cs336_basics/           # Core model implementation (Tokenizer, Transformer)
├── scripts/                # Scripts for training, data prep, and experiments
├── tests/                  # Comprehensive unit tests (Pytest)
├── writeup/                # Experiment reports and W&B charts
├── pyproject.toml          # Project dependencies (uv)
└── cs_spring2025_assignment1_basics.pdf  # Assignment handout

```

---

## 🛠️ Setup & Usage

This project uses [uv](https://github.com/astral-sh/uv) for fast and reliable dependency management.

1. **Install dependencies**:
```bash
uv sync

```


2. **Run tests**:
```bash
uv run pytest

```


3. **Run training**:
```bash
uv run python scripts/train.py

```



---

## 📈 Key Learning Outcomes

* **Architectural Depth:** Mastered the mathematical foundations and coding of Transformer components (Attention, Norms).
* **Data Engineering:** Processed `TinyStories` and `OpenWebText` datasets, gaining insights into subword tokenization.
* **Training Stability:** Gained hands-on experience in debugging loss spikes and tuning hyperparameters through systematic sweeps.

---

## 🔗 Links

* **Course Website:** [Stanford CS336](https://stanford-cs336.github.io/spring2025/)
* **Template Source:** [LCPU-Club Basics](https://github.com/lcpu-club/llm-from-scratch-assignment1-basics)

---

# 中文版本 | Chinese Version

本仓库包含我对 **斯坦福 CS336: 从零开始的语言模型 (2025春季)** 课程作业的个人实现。该项目旨在从底层构建、训练并优化大语言模型。

> [!IMPORTANT]
> **学术诚信提示**：本项目仅用于个人学习和作品展示。如果你是 CS336 的在校学生，请务必遵守所在学校关于代码公开的荣誉准则。

---

## 📝 作业 1：基础部分

在第一个作业中，我从零实现了一个基于 Transformer 的语言模型及其配套的分词器。

### 核心实现

* **BPE 分词器**: 实现了字节对编码（BPE）的训练、编码与解码逻辑。
* **Transformer 架构**: 包含多头自注意力、RMSNorm、SwiGLU 激活函数以及旋转位置编码 (RoPE)。
* **训练流水线**: 实现了带有 AdamW 优化器、梯度裁剪和学习率调度的完整训练循环。

### 🚀 实验报告 (Writeup)

关于实验过程、超参数搜索（LR Sweep）以及模型表现的详细分析，请参阅：
👉 **[作业 1 实验报告](./writeup/writeup.md)**

---

## 📂 仓库结构

```text
.
├── cs336_basics/           # 核心代码 (分词器、Transformer)
├── scripts/                # 训练、数据准备及实验脚本
├── tests/                  # 单元测试 (Pytest)
├── writeup/                # 实验报告与 W&B 图表
├── pyproject.toml          # 依赖配置 (uv)
└── cs_spring2025_assignment1_basics.pdf  # 作业说明文档

```

---

## 🛠️ 环境配置

本项目使用 [uv](https://github.com/astral-sh/uv) 进行包管理。

1. **同步环境**: `uv sync`
2. **运行测试**: `uv run pytest`
3. **开始训练**: `uv run python scripts/train.py`

---

## 📈 核心收获

* **架构深度**: 深入掌握了 Transformer 各组件（如 Attention 和 LayerNorm）的数学原理与实现。
* **数据工程**: 通过处理 `TinyStories` 等数据集，理解了子词分词的细节。
* **训练实战**: 通过系统性的实验，积累了处理训练不稳定性和超参数调优的经验。

