

# CS336 Assignment 1 Writeup: 

# Section 2 

## Problem (unicode1): Understanding Unicode

**(a)** `chr(0)` returns the **Null character** (often denoted as `\0` or `NUL`).

**(b)** Its string representation (`__repr__`) is typically shown as the escaped sequence `'\x00'`, whereas its printed representation is invisible (it prints nothing) or sometimes appears as a placeholder symbol depending on the terminal.

**(c)** In Python strings, `chr(0)` behaves like any other character and does not terminate the string (unlike in C strings). It has a length of 1 and can be concatenated freely (e.g., `"a" + chr(0) + "b"` results in a string of length 3).

## Problem (unicode2): Unicode Encodings

**(a)** We prefer UTF-8 because it is **space-efficient** for ASCII-heavy text (1 byte per char), which dominates source code and English web text. Unlike UTF-32 (fixed 4 bytes), it avoids wasting memory. Unlike UTF-16, it is byte-oriented and has no endianness ambiguity. It is also the standard encoding for the vast majority of the web.

**(b)** The function is incorrect because it attempts to decode the byte string **byte-by-byte** independently. UTF-8 uses variable-length sequences (1-4 bytes) to represent characters.

* **Example**: The character "é" is encoded as `b'\xc3\xa9'`.
* **Explanation**: The function iterates and tries to decode `b'\xc3'` (invalid start byte in isolation) and `b'\xa9'` (continuation byte) separately, causing errors or garbage output, instead of decoding them together as one character.
**(c)** **Example**: `b'\xff'` or `b'\x80'` (a lone continuation byte). Neither of these can appear as a valid valid start of a UTF-8 character sequence.

## Problem (train_bpe_tinystories): BPE Training on TinyStories

**(a)**

* **Training Time**: Training took approximately **146.65 seconds** (~2.5 minutes) on the CPU.
* **Memory Usage**: Peak memory usage was constrained by the map-reduce chunk size (approx. 10MB buffers), staying well within the system limits (e.g., < 16GB).
* **Longest Token**: The longest token is ` accomplishment` (length 15).
* **Sense Check**: Yes, this makes sense. TinyStories contains simple, narrative English text, so long tokens tend to be common, relatively long English words rather than noise or complex compounds.

**(b)**

* **Profiling**: The most time-consuming part of the process is the **Pre-tokenization step** (specifically the regex matching via `re.finditer` or `re.findall`), followed by the iterative counting of pairs in the BPE merge loop if not fully optimized with caching.

## Problem (train_bpe_expts_owt): BPE Training on OpenWebText

**(a)**

* **Longest Token**: The longest token is `b'\xc3\x83\xc3\x82\xc3\x83...'` which decodes to the repeated string `ÃÂÃÂ...` (Length 64).
* **Sense Check**: Yes, this makes sense for web data. This pattern represents **mojibake** (encoding errors) often found in scraped web text (e.g., UTF-8 bytes being misinterpreted as Windows-1252 and then re-encoded). Since this specific noise pattern repeats frequently in the corpus, the BPE algorithm efficiently merged it into a single token.

**(b)**

* **Comparison**: The TinyStories tokenizer learned standard English vocabulary suited for simple narratives. In contrast, the OWT tokenizer learned a much more diverse vocabulary, including code snippets, special formatting characters, foreign language fragments, and common web noise (like the mojibake above). This reflects the higher entropy and "messiness" of the OpenWebText dataset.

## Problem (tokenizer_experiments): Experiments with tokenizers

**(a)**
We measured the compression ratio () on sampled documents:

* **TinyStories Tokenizer (on TinyStories)**: **4.18** bytes/token.
* **OpenWebText Tokenizer (on OpenWebText)**: **4.62** bytes/token.

**(b)**

* **Cross-Domain Performance**: When tokenizing OpenWebText with the TinyStories tokenizer, the compression ratio dropped to **3.33** bytes/token.
* **Observation**: The ratio significantly decreased (worse compression). Qualitatively, the TinyStories tokenizer lacks the specialized vocabulary for web text (like code keywords, complex nouns, or formatting symbols). It is forced to split these unknown words into many small sub-word units or raw bytes, drastically increasing the sequence length.

**(c)**

* **Throughput**: Based on the encoding logs, the tokenizer processed the 30GB OpenWebText dataset in approximately 20 minutes.
* Estimated Throughput .


* **The Pile (825GB)**: At this rate, tokenizing the Pile would take:



**(d)**

* **Data Type Choice**: `uint16` is appropriate because our vocabulary sizes are **10,000** and **32,000**, both of which fit comfortably within the range of an unsigned 16-bit integer ( to ). Using `uint16` instead of `int32` or `int64` cuts the memory and disk usage by 2x or 4x respectively, which is crucial when handling datasets with billions of tokens (saving gigabytes of RAM/disk).


# Section 3: Transformer Language Model Architecture

## Problem (transformer_accounting): Transformer LM resource accounting

### (a) Parameters and Memory

**Configuration (GPT-2 XL):**
* Vocab size ($V$): 50,257
* Context length ($L$): 1,024
* Layers ($N$): 48
* $d_{model}$ ($d$): 1,600
* $d_{ff}$: 6,400
* Heads ($h$): 25 (implies $d_k = 1600/25 = 64$)

**Parameter Count Calculation:**
Based on the assignment architecture (SwiGLU, RoPE, No Bias):

1.  **Embeddings:**
    * Token Embeddings: $V \times d = 50,257 \times 1,600 = 80,411,200$.
    * Position Embeddings (RoPE): $0$ parameters (fixed sinusoidal).
2.  **Transformer Block (per layer):**
    * **Attention Weights (No Bias):**
        * $W_q, W_k, W_v, W_o$: Each is $d \times d$.
        * Count: $4 \times d^2 = 4 \times 1,600^2 = 10,240,000$.
    * **Feed-Forward Weights (SwiGLU, No Bias):**
        * Unlike standard GPT-2 (GELU) which has 2 matrices, SwiGLU has 3 matrices ($W_1, W_2, W_3$).
        * $W_1, W_3$: Size $d \times d_{ff}$. $W_2$: Size $d_{ff} \times d$.
        * Count: $3 \times d \times d_{ff} = 3 \times 1,600 \times 6,400 = 30,720,000$.
    * **Normalization (RMSNorm):**
        * Parameters: $d$ (gain only, no bias).
        * 2 per layer (Pre-norm): $2 \times 1,600 = 3,200$.
    * *Layer Total*: $\approx 40,963,200$.
3.  **Output Layer:**
    * Final RMSNorm: $1,600$.
    * Output Head (assuming untied): $d \times V = 80,411,200$.

**Total Trainable Parameters:**
$$\text{Total} = \text{Embed} + N \times \text{Block} + \text{Head}$$
$$80.4M + 48 \times 40.96M + 80.4M \approx 160.8M + 1,966M \approx \mathbf{2.13 \text{ Billion}}$$

**Memory Requirement (FP32):**
Assuming single-precision (4 bytes per parameter):
$$2.127 \times 10^9 \text{ params} \times 4 \text{ bytes/param} \approx 8.51 \times 10^9 \text{ bytes} \approx \mathbf{8.51 \text{ GB}}$$

---

### (b) Matrix Multiplies and FLOPs

We calculate the floating-point operations (FLOPs) for a single forward pass with batch size $B$ and sequence length $L$. We use the rule that a matrix multiply of $(m \times n)$ and $(n \times p)$ takes $2mnp$ FLOPs.

**1. Attention Mechanism (per layer):**
* **Projections ($W_q, W_k, W_v$):**
    * Input: $(B, L, d)$, Weights: $(d, d)$.
    * $3 \times 2 B L d^2 = 6 B L d^2$.
* **Attention Scores ($QK^T$):**
    * $(B, h, L, d_k) \times (B, h, d_k, L) \rightarrow (B, h, L, L)$.
    * $2 B h L^2 d_k = 2 B L^2 d$ (since $h \times d_k = d$).
* **Weighted Sum ($AV$):**
    * $(B, h, L, L) \times (B, h, L, d_k) \rightarrow (B, h, L, d_k)$.
    * $2 B h L^2 d_k = 2 B L^2 d$.
* **Output Projection ($W_o$):**
    * Input: $(B, L, d)$, Weight: $(d, d)$.
    * $2 B L d^2$.
* ***Attn Total***: $8 B L d^2 + 4 B L^2 d$.

**2. Feed-Forward Network (SwiGLU) (per layer):**
* **Gate & Value ($W_1, W_3$):**
    * Input: $(B, L, d)$, Weights: $(d, d_{ff})$.
    * $2 \times 2 B L d d_{ff} = 4 B L d d_{ff}$.
* **Output ($W_2$):**
    * Input: $(B, L, d_{ff})$, Weight: $(d_{ff}, d)$.
    * $2 B L d d_{ff}$.
* ***FFN Total***: $6 B L d d_{ff}$.

**3. Logits Calculation:**
* Input: $(B, L, d)$, Weight: $(d, V)$.
* $2 B L d V$.

**Total FLOPs Expression:**
$$\text{FLOPs} \approx N \cdot B \cdot L \cdot (8d^2 + 6 d d_{ff} + 4Ld) + 2 B L d V$$

**Total FLOPs (Numerical for GPT-2 XL per token, dividing by $B \cdot L$):**
* $d=1600, d_{ff}=6400, L=1024, N=48, V=50257$.
* Per layer weights ($8d^2 + 6dd_{ff}$): $8(1600^2) + 6(1600)(6400) \approx 20.5M + 61.4M = 81.9M$.
* Per layer attention ($4Ld$): $4(1024)(1600) \approx 6.55M$.
* Logits ($2dV$): $2(1600)(50257) \approx 160.8M$.
* **Total**: $48 \times (81.9M + 6.55M) + 160.8M \approx 4245M + 160.8M \approx \mathbf{4.4 \text{ GFLOPs per token}}$.

---

### (c) Dominant Component

Based on the analysis in (b), the **Feed-Forward Networks (FFN)** require the most FLOPs.
* Within the Transformer blocks, the FFN term is $6 d d_{ff}$. Since $d_{ff} = 4d$ (in this config), this equals $24 d^2$.
* The Attention weights term is $8 d^2$.
* The FFN consumes roughly **3x more compute** than the attention projections and significantly more than the sequence-length dependent operations at this context length ($L=1024$).

---

### (d) Scaling Analysis (Small to Large)

We compare the proportion of FLOPs allocated to the linear projections (Weights) versus the Attention Mechanism (Seq Len dependent) as model size ($d$) increases, keeping $L=1024$ fixed.

* **Weights Term**: $\propto d^2$ (Specifically $32 d^2$ per layer if $d_{ff}=4d$).
* **Attention Term**: $\propto d$ (Specifically $4 L d$ per layer).

**Comparison:**
* **GPT-2 Small** ($d=768$): Ratio of Weights to Attn $\approx \frac{32(768^2)}{4(1024)(768)} = \frac{8 \times 768}{1024} = 6$.
* **GPT-2 XL** ($d=1600$): Ratio of Weights to Attn $\approx \frac{32(1600^2)}{4(1024)(1600)} = \frac{8 \times 1600}{1024} = 12.5$.

**Conclusion:**
As the model size ($d_{model}$) increases, the **Linear Layers (FFN and Projections)** take up a proportionally larger share of the total FLOPs. The sequence-length dependent attention mechanism becomes relatively cheaper (proportionally) for larger models, assuming context length remains constant.

---

### (e) Increasing Context Length

If we increase the context length to **16,384** (a 16x increase):

* The **Attention Mechanism** term ($4 B L^2 d$) grows quadratically with $L$.
* Per token, the cost is $4 L d$.
    * At $L=1024$: $4 \times 1024 \times 1600 \approx 6.5 \text{ MFLOPs}$.
    * At $L=16384$: $4 \times 16384 \times 1600 \approx 105 \text{ MFLOPs}$.
* The **Weights** term (FFN + Projections) is constant per token at $\approx 82 \text{ MFLOPs}$.

**Conclusion:**
At $L=16,384$, the **Self-Attention mechanism** surpasses the FFNs to become the dominant computational component. The relative contribution of the $O(L^2)$ attention operations shifts from being negligible to being the primary bottleneck.

# Section 4: Training a Transformer LM

## Problem (learning_rate_tuning): Tuning the learning rate

**Observations:**
I ran the SGD toy example with three learning rates (0.1, 0.01, 0.001) for 10 iterations. The results were:
* **LR = 0.1**: The loss decayed the **fastest**, dropping significantly from 24.17 to **23.72**.
* **LR = 0.01**: The loss decayed **slower**, ending at **24.12**.
* **LR = 0.001**: The loss decayed the **slowest**, showing negligible progress (ending at **24.16**).

**Conclusion:**
In this convex toy problem, the largest learning rate (0.1) provided the best convergence speed without diverging. The smallest learning rate (0.001) was too conservative to make meaningful progress within the limited step count .

---

## Problem (adamwAccounting): Resource accounting for training with AdamW

### (a) Peak Memory Usage Expressions
We assume all tensors are stored in `float32` (4 bytes per element).

1.  **Parameters ($P_{bytes}$)**:
    $$P_{bytes} = 4 \times P$$
    Where $P$ is the total number of trainable parameters.

2.  **Gradients ($G_{bytes}$)**:
    We store gradients for every parameter.
    $$G_{bytes} = 4 \times P$$

3.  **Optimizer State ($O_{bytes}$)**:
    AdamW maintains two state tensors per parameter: the first moment $m$ and the second moment $v$.
    $$O_{bytes} = 4 \times (P + P) = 8 \times P$$

4.  **Activations ($A_{bytes}$)**:
    Based on the components listed in the handout, we sum the sizes of tensors that must be stored for the backward pass (per batch item, for $N$ layers, context length $L$, dimension $d$, heads $h$, vocab $V$, and $d_{ff} = 4d$).

    * **Per Transformer Block**:
        * RMSNorm inputs: $L \times d$
        * **Multi-head Attention**:
            * QKV Output: $L \times 3d$
            * Attention Scores ($QK^T$): $h \times L \times L$
            * Softmax Probabilities: $h \times L \times L$
            * Output Projection Input: $L \times d$
        * **Feed-Forward ($d_{ff}=4d$)**:
            * $W_1$ output (input to SiLU): $L \times 4d$
            * SiLU output (input to $W_2$): $L \times 4d$
            * Residual inputs (stored at block start/mid): $\approx 2 \times L \times d$
        * *Block Total (elements)*: $12Ld + 2hL^2$.
    * **Final Layers**:
        * Final RMSNorm: $L \times d$
        * Logits: $L \times V$
    
    $$A_{total\_elements} = B \times [ N(12Ld + 2hL^2) + Ld + LV ]$$
    $$A_{bytes} = 4 \times A_{total\_elements}$$

**Total Memory**: $16 P + A_{bytes}$ (bytes).

### (b) GPT-2 XL Instantiation
**Configuration**: $V=50257, L=1024, N=48, d=1600, h=25, d_{ff}=6400$.

1.  **Static Memory (Model + Optimizer)**:
    For GPT-2 XL (standard FFN), $P \approx 1.6$ Billion parameters.
    $$\text{Static} = 16 \text{ bytes} \times 1.6 \times 10^9 \approx 25.6 \text{ GB}$$

2.  **Activations (per batch)**:
    * Per Layer: $12(1024)(1600) + 2(25)(1024)^2 \approx 19.6M + 52.4M = 72 \text{ M elements}$.
    * All Layers ($N=48$): $48 \times 72M \approx 3.45 \text{ B elements}$.
    * Logits: $1024 \times 50257 \approx 51.5 \text{ M elements}$.
    * Total Elements: $\approx 3.5 \text{ Billion}$.
    * Memory per batch: $3.5 \times 10^9 \times 4 \text{ bytes} \approx 14 \text{ GB}$.

**Expression:**
$$\text{Memory (GB)} \approx 25.6 + 14 \times \text{batch\_size}$$

**Maximum Batch Size (80GB limit):**
$$25.6 + 14 \times B \le 80$$
$$14 \times B \le 54.4$$
$$B \le 3.88$$
Thus, the maximum batch size is **3**.

### (c) AdamW FLOPs
For each parameter, AdamW performs the following element-wise operations:
1.  Update $m$: 3 FLOPs (mul, add, mul)
2.  Update $v$: 4 FLOPs (mul, add, mul, square)
3.  Update $\theta$: $\approx 5$ FLOPs (sqrt, add, div, mul, sub)
4.  Weight decay: 2 FLOPs (mul, sub)

[cite_start]**Total**: Approximately **14 FLOPs per parameter** per step [cite: 914-922].

### (d) Training Time on A100 (Float32)
* **Total Training FLOPs ($C_{total}$)**:
    * Model size $P \approx 1.6 \times 10^9$.
    * Tokens processed: $B \times L \times \text{steps} = 1024 \times 1024 \times 400,000 \approx 4.2 \times 10^{11}$ tokens.
    * Approximate FLOPs per token (Fwd + Bwd): $6 \times P$.
    * $C_{total} = 6 \times 1.6 \times 10^9 \times 4.2 \times 10^{11} \approx 4.03 \times 10^{21}$ FLOPs.
* **Hardware Throughput**:
    * [cite_start]Theoretical Peak (Float32): 19.5 TFLOPs[cite: 952].
    * Achieved MFU (50%): $0.5 \times 19.5 = 9.75$ TFLOPs ($9.75 \times 10^{12}$ FLOPs/s).
* **Time**:
    $$\text{Time (s)} = \frac{4.03 \times 10^{21}}{9.75 \times 10^{12}} \approx 4.13 \times 10^8 \text{ seconds}$$
    $$\text{Time (days)} = \frac{4.13 \times 10^8}{3600 \times 24} \approx \mathbf{4,784 \text{ days}}$$

*Note: This surprisingly high number illustrates why modern LLM training relies on Tensor Cores (BF16/FP16) which offer ~312 TFLOPs on A100s, rather than the standard Float32 CUDA cores cited in the problem prompt.*

Based on the handout, **Section 5 (Training Loop)** focuses entirely on code implementation and does not contain specific theoretical questions (like "What is the complexity...") that require written answers in the `writeup.pdf`. The "Deliverables" for this section are functions and scripts.

However, it is good practice to include a brief summary of your implementation details in your writeup to demonstrate completion. Below is a structured summary of the Section 5 implementation in Markdown format, which you can copy into your report.

---

# 5 Training Loop

### Problem (training_together)

**Implementation Details:**
I implemented the complete training script in `train.py` that integrates the model, optimizer, scheduler, and data loader.


**Configuration:** The script uses `argparse` to allow full control over hyperparameters (e.g., `batch_size`, `lr`, `vocab_size`) and system settings (`device`, `out_dir`).



**Large Dataset Support:** Training and validation data are loaded using `np.memmap` to handle the memory constraints of large pretraining corpora.



**Checkpointing:** The loop periodically serializes the model and optimizer state to the user-provided path.



**Logging:** The script logs training loss, learning rate, and throughput (tokens/sec) to the console and optionally supports 

**Weights and Biases (WandB)** for external visualization of learning curves.



# 6 Generating Text

### Problem (decoding)

**Implementation Details:**

I implemented the generation logic in the `TransformerLM.generate` method. The function takes a conditioning sequence `idx` and generates `max_new_tokens` autoregressively. The implementation supports the following features as required:

1. **Context Management:**
At each step, if the input sequence length exceeds the model's `context_length` (e.g., 256), I truncate the input to keep only the last `context_length` tokens to ensure the positional embeddings remain valid.


2. **Temperature Scaling:**
I implemented temperature scaling by dividing the logits by the `temperature` parameter before applying softmax.
* If `temperature > 0`, the logits are scaled as .


* If `temperature` approaches 0 (or is explicitly set to 0 in my code), the logic falls back to greedy decoding using `torch.topk(..., k=1)` to select the token with the highest probability.




3. **Top-p (Nucleus) Sampling:**
I implemented Nucleus Sampling following Holtzman et al. (2020) .


* The probabilities are sorted in descending order.
* A cumulative sum is computed to identify the smallest set of tokens whose cumulative probability exceeds the threshold `top_p`.
* A boolean mask is created to zero out (set to `-inf`) the logits of all tokens outside this set before sampling.




4. **Stop Condition:**
The generation loop checks if the sampled token matches the `<|endoftext|>` ID (`eos_id`). If it does, the generation loop terminates early to prevent generating meaningless content after the end of the document.



**Script:**
I created a standalone script `scripts/generate.py` that loads the trained model checkpoint and tokenizer, accepts a user prompt, and outputs the generated text to the console.

**Sample Output:**
Using the model trained on TinyStories (approx. 5000 steps) with `temperature=0.7`, `top_p=0.9`, and the prompt "Once upon a time":

> Once upon a time, there was a little girl named Amy. She was a very obedient girl who always listened to her mom and dad. One day, her mom said, "Amy, you need to be very careful with the things you love." Amy did not listen. She wanted to listen to her mom and dad.
<|endoftext|>

# Section 7: Experiments

## 7.2 Hyperparameter Tuning

### Problem (learning_rate)

**Deliverable: Learning curve comparison**

![alt text](<W&B Chart 2026_1_28 00_01_18.png>)

**Analysis:**

To identify the optimal learning rate and understand the stability boundaries of the model, I performed a sweep over the values . The results are analyzed as follows:

1. **Under-confidence ():** The pink curve shows the slowest convergence. By step 5000, the loss is still around **2.1**, significantly higher than well-tuned models. The step size is too small to traverse the loss landscape efficiently within the compute budget.
2. **Optimal Performance ( & ):**
* The **Red** learning rate appears to be the most effective, achieving the fastest initial descent and the lowest final validation loss ( **1.60**).
* The **(Green)** rate is also highly effective and stable, closely trailing the optimal performance.


3. **The Edge of Stability ():**
* The **(Blue)** curve demonstrates the concept of the "Edge of Stability." While it did not cause the model to diverge to NaN, it performed worse ( 1.7) than the  run.
* This indicates that the learning rate is slightly too high, preventing the model from settling into the sharpest minima and likely causing it to oscillate around the optimal solution. The optimal rate is found just before this point of degradation.



---

### Problem (batch_size_experiment)

**Deliverable: Learning curve comparison**

![alt text](<W&B Chart 2026_1_28 00_24_59.png>)

**Analysis:**

I compared training with a batch size of 1 versus a batch size of 64 (keeping other hyperparameters constant).

* **Batch Size 64 (Red/Orange):** This configuration is stable and efficient. The gradients are estimated over a larger number of samples, providing a lower-variance estimate of the true gradient. This allows the model to take consistent steps toward the minimum, resulting in a smooth loss curve and a low final loss ( 1.6).
* **Batch Size 1 (Pink):** This configuration performs poorly. The loss curve is extremely jagged and noisy (High Variance). Because the gradient is estimated from a single sample, it contains significant noise, causing the optimization trajectory to be chaotic. The final loss ( 2.9) is much higher, proving that small batch sizes (without gradient accumulation) lead to poor convergence stability.

---

## 7.3 Ablation Studies

### Problem (ablation)

**Deliverable: Learning curve comparison of architecture changes**

![alt text](<W&B Chart 2026_1_28 00_26_39.png>)

**Analysis:**

I compared the "Standard Baseline" against several architectural variations. The results at step 5000 highlight the importance of each component:

1. **Baseline (Pre-Norm, RMSNorm, SwiGLU, RoPE):** This configuration yields the best performance (Loss  1.6 - 1.7), confirming that the modern Transformer recipe is highly effective.
2. **No RMSNorm (`ablation_no_rmsnorm`):** This was the most detrimental change (Loss  2.2). Without Layer Normalization, the network suffers from internal covariate shift and potentially exploding/vanishing gradients, making deep training significantly harder.
3. **No Positional Embeddings (`ablation_no_rope`):** Removing RoPE degraded performance (Loss  1.85). Without explicit positional encoding, the self-attention mechanism is permutation invariant and acts like a "bag of words," losing critical sequential structure required for language modeling.
4. **SiLU instead of SwiGLU (`ablation_silu`):** Replacing SwiGLU with a standard SiLU activation resulted in a higher loss ( 1.8). This validates findings (e.g., from PaLM/LLaMA) that Gated Linear Units provide better inductive biases and capacity for LLMs.
5. **Post-Norm (`ablation_post_norm`):** This performed similarly to or slightly worse than the Pre-Norm baseline. While Post-Norm was the original design, Pre-Norm is generally preferred in modern LLMs for better gradient flow and training stability without warm-up.

---

## 7.4 OpenWebText Experiment

### Problem (main_experiment)

**Deliverable: Learning curve comparison**

![alt text](<W&B Chart 2026_1_28 00_01_18-1.png>)

**Comparison of Losses:**

* **TinyStories Baseline Loss:**  1.6 - 1.7
* **OpenWebText (OWT) Loss:**  4.8

**Interpretation:**
The loss on OpenWebText is drastically higher than on TinyStories, but this **does not** imply a model bug. It reflects the fundamental difference in **Data Entropy**:

* **TinyStories** has a vocabulary of only ~10k and uses simple, repetitive grammar suitable for children. It has low information entropy, making prediction easy.
* **OpenWebText** represents the complexity of the real internet (>50k vocabulary, complex syntax, code, diverse topics). The uncertainty (entropy) of the next token is naturally much higher, leading to a higher cross-entropy floor.

**Deliverable: Generated Text Analysis**

**Generated Text Sample:**

> Once upon a time of my life, I started talking about something else I had to talk to for myself and talk to me about my work. I have always wanted to write my life, because I was like, ‘I have a life.’
<|endoftext|>

**Fluency & Quality Analysis:**
The generated text from the OWT model is largely **incoherent**. While it produces valid English words, it fails to form meaningful sentences or maintain a consistent topic.

**Why is it worse than TinyStories?**
Despite using the same architecture and training iterations:

1. **Underfitting (Capacity Gap):** A ~20M parameter model is sufficient to memorize the simple patterns of TinyStories, but it is woefully insufficient to model the vast complexity of the OpenWebText distribution.
2. **Insufficient Training:** 5000 steps on OWT (which is hundreds of GBs) means the model has seen a negligible fraction of the dataset. It has not converged, whereas the TinyStories model has likely seen its smaller dataset multiple times (or a significant portion of it). The model is severely undertrained for this data distribution.