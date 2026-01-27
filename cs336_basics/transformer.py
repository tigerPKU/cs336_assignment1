import torch
import torch.nn as nn
import math


class Linear(nn.Module):
    def __init__(self, in_features, out_features, bias=False):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.trunc_normal_(self.weight, std=0.02)

    def forward(self, x):
        return x @ self.weight.T


class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim))
        nn.init.trunc_normal_(self.weight, std=1.0)

    def forward(self, x):
        return self.weight[x]


class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        x_fp32 = x.float()
        rms = torch.sqrt(x_fp32.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (x_fp32 / rms).type_as(x) * self.weight


def silu(x):
    return x * torch.sigmoid(x)


# === 修改点 1: 支持 SiLU 和 SwiGLU 切换 ===
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, ffn_type="swiglu"):
        super().__init__()
        self.ffn_type = ffn_type
        self.w1 = Linear(d_model, d_ff, bias=False)
        self.w2 = Linear(d_ff, d_model, bias=False)

        if ffn_type == "swiglu":
            self.w3 = Linear(d_model, d_ff, bias=False)
        # 如果是 silu，不需要 w3

    def forward(self, x):
        if self.ffn_type == "swiglu":
            return self.w2(silu(self.w1(x)) * self.w3(x))
        elif self.ffn_type == "silu":
            return self.w2(silu(self.w1(x)))
        else:
            raise ValueError(f"Unknown ffn_type: {self.ffn_type}")


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    max_val = torch.max(x, dim=dim, keepdim=True).values
    x_safe = x - max_val
    exp_x = torch.exp(x_safe)
    sum_exp = torch.sum(exp_x, dim=dim, keepdim=True)
    return exp_x / sum_exp


def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    if logits.ndim == 3:
        B, T, V = logits.shape
        logits = logits.view(B * T, V)
        targets = targets.view(B * T)
    max_val = logits.max(dim=-1, keepdim=True).values
    logits_stable = logits - max_val
    log_sum_exp = torch.log(torch.sum(torch.exp(logits_stable), dim=-1, keepdim=True))
    log_probs = logits_stable - log_sum_exp
    N = logits.shape[0]
    target_log_probs = log_probs[torch.arange(N, device=logits.device), targets]
    return -target_log_probs.mean()


class RotaryPositionalEmbedding(torch.nn.Module):
    def __init__(self, d_model: int, max_seq_len: int, theta: float = 10000.0):
        super().__init__()
        dim = d_model
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq_len).float()
        freqs_outer = torch.outer(t, freqs)
        freqs_interleaved = freqs_outer.repeat_interleave(2, dim=-1)
        self.register_buffer("cos_cached", freqs_interleaved.cos().type(torch.float32))
        self.register_buffer("sin_cached", freqs_interleaved.sin().type(torch.float32))

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor):
        cos = self.cos_cached[token_positions]
        sin = self.sin_cached[token_positions]
        if x.ndim == 4 and cos.ndim == 3:
            cos = cos.unsqueeze(1)
            sin = sin.unsqueeze(1)
        elif x.ndim == 3 and cos.ndim == 3 and x.shape[0] != cos.shape[0]:
            if x.shape[0] % cos.shape[0] == 0:
                ratio = x.shape[0] // cos.shape[0]
                cos = cos.unsqueeze(1).expand(-1, ratio, -1, -1).reshape(x.shape)
                sin = sin.unsqueeze(1).expand(-1, ratio, -1, -1).reshape(x.shape)
        return (x * cos) + (self._rotate_adjacent(x) * sin)

    def _rotate_adjacent(self, x: torch.Tensor):
        original_shape = x.shape
        x_reshaped = x.view(*original_shape[:-1], -1, 2)
        rotate_x = torch.stack((-x_reshaped[..., 1], x_reshaped[..., 0]), dim=-1)
        return rotate_x.flatten(-2)


# === 修改点 2: 传入 use_rope 参数 ===
class MultiHeadSelfAttention(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, max_seq_len: int, theta: float = 10000.0, use_rope: bool = True):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.use_rope = use_rope

        self.q_proj = Linear(d_model, d_model)
        self.k_proj = Linear(d_model, d_model)
        self.v_proj = Linear(d_model, d_model)
        self.output_proj = Linear(d_model, d_model)

        if use_rope:
            self.rope = RotaryPositionalEmbedding(self.head_dim, max_seq_len, theta)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor = None):
        batch_size, seq_len, _ = x.shape
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        if self.use_rope:
            if token_positions is None:
                token_positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, seq_len)
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        scores.masked_fill_(mask, -float("inf"))

        attn_probs = softmax(scores, dim=-1)
        context = (attn_probs @ v).transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.output_proj(context)


# === 修改点 3: 支持 Pre/Post Norm 和 LayerNorm Ablation ===
class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model,
        num_heads,
        d_ff,
        max_seq_len,
        theta=10000.0,
        use_rms_norm=True,
        norm_mode="pre",
        ffn_type="swiglu",
        use_rope=True,
    ):
        super().__init__()
        self.use_rms_norm = use_rms_norm
        self.norm_mode = norm_mode

        if use_rms_norm:
            self.ln1 = RMSNorm(d_model)
            self.ln2 = RMSNorm(d_model)

        self.attn = MultiHeadSelfAttention(d_model, num_heads, max_seq_len, theta, use_rope=use_rope)
        self.ffn = FeedForward(d_model, d_ff, ffn_type=ffn_type)

    def forward(self, x):
        if self.norm_mode == "pre":
            # Pre-Norm: x = x + f(norm(x))
            # 如果不使用 norm，则 x = x + f(x)

            # Sublayer 1: Attention
            residual = x
            if self.use_rms_norm:
                x = self.ln1(x)
            x = residual + self.attn(x)

            # Sublayer 2: FFN
            residual = x
            if self.use_rms_norm:
                x = self.ln2(x)
            x = residual + self.ffn(x)

        elif self.norm_mode == "post":
            # Post-Norm: x = norm(x + f(x))

            # Sublayer 1: Attention
            x = x + self.attn(x)
            if self.use_rms_norm:
                x = self.ln1(x)

            # Sublayer 2: FFN
            x = x + self.ffn(x)
            if self.use_rms_norm:
                x = self.ln2(x)

        return x


# === 修改点 4: 将配置参数传入 Block ===
class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model,
        num_layers,
        num_heads,
        d_ff,
        max_seq_len,
        theta=10000.0,
        use_rms_norm=True,
        norm_mode="pre",
        ffn_type="swiglu",
        use_rope=True,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.token_embeddings = Embedding(vocab_size, d_model)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model,
                    num_heads,
                    d_ff,
                    max_seq_len,
                    theta,
                    use_rms_norm=use_rms_norm,
                    norm_mode=norm_mode,
                    ffn_type=ffn_type,
                    use_rope=use_rope,
                )
                for _ in range(num_layers)
            ]
        )

        if use_rms_norm:
            self.ln_final = RMSNorm(d_model)
        else:
            self.ln_final = nn.Identity()

        self.lm_head = Linear(d_model, vocab_size, bias=False)

    def forward(self, x):
        x = self.token_embeddings(x)
        for layer in self.layers:
            x = layer(x)
        x = self.ln_final(x)
        return self.lm_head(x)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_p=None, eos_id=None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.max_seq_len else idx[:, -self.max_seq_len :]
            logits = self(idx_cond)
            logits = logits[:, -1, :]

            if temperature > 0.0:
                logits = logits / temperature
            else:
                _, idx_next = torch.topk(logits, k=1, dim=-1)
                idx = torch.cat((idx, idx_next), dim=1)
                if eos_id is not None and (idx_next == eos_id).all():
                    break
                continue

            if top_p is not None and top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                sorted_probs = torch.softmax(sorted_logits, dim=-1)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float("-inf")

            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

            if eos_id is not None and (idx_next == eos_id).all():
                break

        return idx


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p.data)
                    state["exp_avg_sq"] = torch.zeros_like(p.data)
                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                state["step"] += 1
                t = state["step"]
                grad = p.grad.data
                p.data.mul_(1 - lr * weight_decay)
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                bias_correction1 = 1 - beta1**t
                bias_correction2 = 1 - beta2**t
                step_size = lr * math.sqrt(bias_correction2) / bias_correction1
                denom = exp_avg_sq.sqrt().add_(eps)
                p.data.addcdiv_(exp_avg, denom, value=-step_size)
        return loss


def gradient_clipping(parameters, max_l2_norm):
    total_norm = 0.0
    for p in parameters:
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm**0.5
    clip_coef = max_l2_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for p in parameters:
            if p.grad is not None:
                p.grad.data.mul_(clip_coef)


def get_lr_cosine_schedule(it, max_learning_rate, min_learning_rate, warmup_iters, cosine_cycle_iters):
    if it < warmup_iters:
        return max_learning_rate * (it / warmup_iters)
    if it > cosine_cycle_iters:
        return min_learning_rate
    decay_ratio = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_learning_rate + coeff * (max_learning_rate - min_learning_rate)
