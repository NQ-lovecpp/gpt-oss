import json
import math
import os
from dataclasses import dataclass

import torch
import torch.distributed as dist

from gpt_oss.torch.weights import Checkpoint


@dataclass
class ModelConfig:
    """模型结构与推理行为的集中配置。

    该 dataclass 对应 checkpoint 中的 `config.json`，用于：
    1) 构建网络拓扑（层数、维度、头数、MoE 专家数等）；
    2) 控制注意力窗口与 RoPE/YaRN 的外推参数；
    3) 保持模型定义与权重文件的一致性。
    """

    # Transformer block 数量。
    num_hidden_layers: int = 36
    # MoE 专家总数。
    num_experts: int = 128
    # 每个 token 路由到的 top-k 专家数。
    experts_per_token: int = 4
    # 词表大小（embedding / unembedding 的 token 维度）。
    vocab_size: int = 201088
    # 主干隐藏维度（残差流维度）。
    hidden_size: int = 2880
    # MLP 中间层总维度（会按并行 world_size 分片）。
    intermediate_size: int = 2880
    # SwiGLU 输入裁剪上限，控制数值稳定性。
    swiglu_limit: float = 7.0
    # 单个注意力头的通道维度。
    head_dim: int = 64
    # Q 头数（总注意力头数）。
    num_attention_heads: int = 64
    # KV 头数（GQA/MQA 风格共享键值头）。
    num_key_value_heads: int = 8
    # 局部注意力窗口大小；0 表示全局注意力。
    sliding_window: int = 128
    # 训练时初始上下文长度（YaRN/NTK 外推参考值）。
    initial_context_length: int = 4096
    # RoPE 频率底数。
    rope_theta: float = 150000.0
    # RoPE 扩展因子（>1 时启用 YaRN 风格缩放）。
    rope_scaling_factor: float = 32.0
    # NTK 分段插值下界参数。
    rope_ntk_alpha: float = 1.0
    # NTK 分段插值上界参数。
    rope_ntk_beta: float = 32.0


class RMSNorm(torch.nn.Module):
    def __init__(
        self, num_features: int, eps: float = 1e-05, device: torch.device | None = None
    ):
        super().__init__()
        # 最后一维的特征数（即要归一化的通道数）。
        self.num_features = num_features
        # 防止除零与极小方差导致的数值不稳定。
        self.eps = eps
        # 可学习缩放参数，维度与特征维一致。
        self.scale = torch.nn.Parameter(
            torch.ones(num_features, device=device, dtype=torch.float32)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 约束输入最后一维必须匹配 RMSNorm 定义的特征数。
        assert x.shape[-1] == self.num_features
        # 用 float32 计算均方根，提升稳定性；最后再还原原始 dtype。
        t, dtype = x.float(), x.dtype
        # RMSNorm: x / sqrt(mean(x^2) + eps)
        t = t * torch.rsqrt(torch.mean(t**2, dim=-1, keepdim=True) + self.eps)
        # 逐通道缩放并回到原 dtype（通常为 bfloat16）。
        return (t * self.scale).to(dtype)


def _apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    # 扩展到 head 维并与输入 dtype 对齐，避免隐式类型转换开销。
    cos = cos.unsqueeze(-2).to(x.dtype)
    sin = sin.unsqueeze(-2).to(x.dtype)
    # 将通道按偶/奇位对半拆分，执行二维旋转。
    x1, x2 = torch.chunk(x, 2, dim=-1)
    # [x1, x2] 乘以旋转矩阵 [[cos, -sin], [sin, cos]]。
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
    # 拼回原通道维。
    return torch.cat((o1, o2), dim=-1)


class RotaryEmbedding(torch.nn.Module):
    def __init__(
        self,
        head_dim: int,
        base: int,
        dtype: torch.dtype,
        initial_context_length: int = 4096,
        scaling_factor: float = 1.0,
        ntk_alpha: float = 1.0,
        ntk_beta: float = 32.0,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        # 每个注意力头的维度（要求可被 2 整除，以便做偶奇配对旋转）。
        self.head_dim = head_dim
        # RoPE 的频率底数。
        self.base = base
        # 预留 dtype 字段，便于将来扩展精度控制（当前实现主要用 float32 计算）。
        self.dtype = dtype
        # 训练上下文长度，用于 YaRN/NTK 的分段外推。
        self.initial_context_length = initial_context_length
        # 长上下文扩展倍率。
        self.scaling_factor = scaling_factor
        # NTK 分段参数 alpha / beta。
        self.ntk_alpha = ntk_alpha
        self.ntk_beta = ntk_beta
        self.device = device

    def _compute_concentration_and_inv_freq(self) -> torch.Tensor:
        """See YaRN paper: https://arxiv.org/abs/2309.00071"""
        freq = self.base ** (
            torch.arange(0, self.head_dim, 2, dtype=torch.float, device=self.device)
            / self.head_dim
        )
        if self.scaling_factor > 1.0:
            # YaRN 给出的 concentration 项：对 cos/sin 振幅做统一缩放。
            concentration = (
                0.1 * math.log(self.scaling_factor) + 1.0
            )  # YaRN concentration

            # 只看半维（因为偶奇两两配对），用于计算分段边界。
            d_half = self.head_dim / 2
            # NTK by parts
            low = (
                d_half
                * math.log(self.initial_context_length / (self.ntk_beta * 2 * math.pi))
                / math.log(self.base)
            )
            high = (
                d_half
                * math.log(self.initial_context_length / (self.ntk_alpha * 2 * math.pi))
                / math.log(self.base)
            )
            assert 0 < low < high < d_half - 1

            # interpolation: 缩放后频率；extrapolation: 原始频率。
            interpolation = 1.0 / (self.scaling_factor * freq)
            extrapolation = 1.0 / freq

            # 线性 ramp 形成软分段掩码，在 [low, high] 平滑过渡。
            ramp = (
                torch.arange(d_half, dtype=torch.float32, device=freq.device) - low
            ) / (high - low)
            mask = 1 - ramp.clamp(0, 1)

            # 混合两套频率，实现 NTK-aware 频率重参数化。
            inv_freq = interpolation * (1 - mask) + extrapolation * mask
        else:
            # 不做长上下文扩展时，退化为标准 RoPE。
            concentration = 1.0
            inv_freq = 1.0 / freq

        return concentration, inv_freq

    def _compute_cos_sin(self, num_tokens: int):
        # 根据当前序列长度生成位置相关的 cos/sin 表。
        concentration, inv_freq = self._compute_concentration_and_inv_freq()
        t = torch.arange(num_tokens, dtype=torch.float32, device=self.device)
        # 外积得到 [position, frequency] 相位矩阵。
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        # YaRN concentration 会统一放大/缩小振幅。
        cos = freqs.cos() * concentration
        sin = freqs.sin() * concentration
        return cos, sin

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # 采用 token-first 布局，第一维是序列长度。
        num_tokens = query.shape[0]
        cos, sin = self._compute_cos_sin(num_tokens)

        # 将 query 末维整理为 (..., head_dim) 以应用旋转，再恢复原形状。
        query_shape = query.shape
        query = query.view(num_tokens, -1, self.head_dim)
        query = _apply_rotary_emb(query, cos, sin)
        query = query.reshape(query_shape)

        # key 同理。
        key_shape = key.shape
        key = key.view(num_tokens, -1, self.head_dim)
        key = _apply_rotary_emb(key, cos, sin)
        key = key.reshape(key_shape)
        return query, key


def sdpa(Q, K, V, S, sm_scale, sliding_window=0):
    # sliding_window == 0 means no sliding window
    # Q: [T, H_kv, q_mult, D]，K/V: [T, H_kv, D]
    # 其中 q_mult = H_q / H_kv，用于 GQA 中一个 KV 头对应多个 Q 头。
    n_tokens, n_heads, q_mult, d_head = Q.shape
    assert K.shape == (n_tokens, n_heads, d_head)
    assert V.shape == (n_tokens, n_heads, d_head)
    # 扩展 K/V 以匹配 q_mult 维度，避免显式复制（expand 视图）。
    K = K[:, :, None, :].expand(-1, -1, q_mult, -1)
    V = V[:, :, None, :].expand(-1, -1, q_mult, -1)
    # S 是每个头的 attention sink，对应 softmax 的额外一列。
    S = S.reshape(n_heads, q_mult, 1, 1).expand(-1, -1, n_tokens, -1)
    # 因果遮罩：禁止看见未来 token。
    mask = torch.triu(Q.new_full((n_tokens, n_tokens), -float("inf")), diagonal=1)
    if sliding_window > 0:
        # 滑动窗口额外屏蔽过远历史，只保留最近 sliding_window 个 token。
        mask += torch.tril(
            mask.new_full((n_tokens, n_tokens), -float("inf")), diagonal=-sliding_window
        )
    # 计算注意力分数，输出布局 [H_kv, q_mult, T_q, T_k]。
    QK = torch.einsum("qhmd,khmd->hmqk", Q, K)
    QK *= sm_scale
    QK += mask[None, None, :, :]
    # 拼接 sink 列并做 softmax。
    QK = torch.cat([QK, S], dim=-1)
    W = torch.softmax(QK, dim=-1)
    # 去掉 sink 列，只保留真实 token 的权重。
    W = W[..., :-1]
    # 按权重聚合 V，最后合并头维返回 [T, H_q * D]。
    attn = torch.einsum("hmqk,khmd->qhmd", W, V)
    return attn.reshape(n_tokens, -1)


class AttentionBlock(torch.nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        layer_idx: int = 0,
        device: torch.device | None = None,
    ):
        super().__init__()
        # 头部超参数缓存。
        self.head_dim = config.head_dim
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        # Only apply sliding window to every other layer
        self.sliding_window = config.sliding_window if layer_idx % 2 == 0 else 0
        # 每个 Q 头一个 sink 参数，用于稳定长序列注意力分布。
        self.sinks = torch.nn.Parameter(
            torch.empty(config.num_attention_heads, device=device, dtype=torch.bfloat16)
        )
        # 预归一化（Pre-Norm）结构。
        self.norm = RMSNorm(config.hidden_size, device=device)
        # 单线性层一次性产出 Q/K/V，减少 kernel 启动与访存。
        qkv_dim = config.head_dim * (
            config.num_attention_heads + 2 * config.num_key_value_heads
        )
        self.qkv = torch.nn.Linear(
            config.hidden_size, qkv_dim, device=device, dtype=torch.bfloat16
        )
        # 输出投影回 residual hidden_size。
        self.out = torch.nn.Linear(
            config.head_dim * config.num_attention_heads,
            config.hidden_size,
            device=device,
            dtype=torch.bfloat16,
        )
        # 标准 attention 缩放因子 1/sqrt(d_head)。
        self.sm_scale = 1 / math.sqrt(config.head_dim)
        # RoPE 模块负责对 Q/K 注入位置信息。
        self.rope = RotaryEmbedding(
            config.head_dim,
            config.rope_theta,
            torch.float32,
            initial_context_length=config.initial_context_length,
            scaling_factor=config.rope_scaling_factor,
            ntk_alpha=config.rope_ntk_alpha,
            ntk_beta=config.rope_ntk_beta,
            device=device,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1) 归一化后做 QKV 投影。
        t = self.norm(x)
        qkv = self.qkv(t)
        # 2) 按预定义切片拆分 q / k / v。
        q = qkv[:, : self.num_attention_heads * self.head_dim].contiguous()
        k = qkv[
            :,
            self.num_attention_heads
            * self.head_dim : (self.num_attention_heads + self.num_key_value_heads)
            * self.head_dim,
        ].contiguous()
        v = qkv[
            :,
            (self.num_attention_heads + self.num_key_value_heads)
            * self.head_dim : (self.num_attention_heads + 2 * self.num_key_value_heads)
            * self.head_dim,
        ].contiguous()

        # 3) 重排为 SDPA 所需形状；q_mult 表示每个 KV 头对应的 Q 头倍数。
        q = q.view(
            -1,
            self.num_key_value_heads,
            self.num_attention_heads // self.num_key_value_heads,
            self.head_dim,
        )
        k = k.view(-1, self.num_key_value_heads, self.head_dim)
        v = v.view(-1, self.num_key_value_heads, self.head_dim)
        # 4) 在 Q/K 上应用旋转位置编码。
        q, k = self.rope(q, k)
        # 5) 执行注意力并做输出投影。
        t = sdpa(q, k, v, self.sinks, self.sm_scale, self.sliding_window)
        t = self.out(t)
        # 6) 残差连接。
        t = x + t
        return t


def swiglu(x, alpha: float = 1.702, limit: float = 7.0):
    # 将最后一维交替拆分：偶位走门控，奇位走线性支路。
    x_glu, x_linear = x[..., ::2], x[..., 1::2]
    # Clamp the input values
    # 门控分支只上限裁剪；线性分支双边裁剪，控制极端值。
    x_glu = x_glu.clamp(min=None, max=limit)
    x_linear = x_linear.clamp(min=-limit, max=limit)
    # SiLU-like 门控：x * sigmoid(alpha * x)。
    out_glu = x_glu * torch.sigmoid(alpha * x_glu)
    # Note we add an extra bias of 1 to the linear layer
    # 与 (x_linear + 1) 相乘，保证初期有更平滑的直通路径。
    return out_glu * (x_linear + 1)


class MLPBlock(torch.nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        device: torch.device | None = None,
    ):
        super().__init__()
        # MoE 路由配置。
        self.num_experts = config.num_experts
        self.experts_per_token = config.experts_per_token
        self.swiglu_limit = config.swiglu_limit
        # 分布式并行信息（未初始化则退化为单卡）。
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        # 与注意力一致的 pre-norm 结构。
        self.norm = RMSNorm(config.hidden_size, device=device)
        # 路由器：为每个 token 输出各专家打分。
        self.gate = torch.nn.Linear(
            config.hidden_size, config.num_experts, device=device, dtype=torch.bfloat16
        )
        # 中间维度按张量并行分片，确保可整除。
        assert config.intermediate_size % self.world_size == 0
        # 第一层专家权重，输出是 2 * intermediate（供 SwiGLU 双分支使用）。
        self.mlp1_weight = torch.nn.Parameter(
            torch.empty(
                (
                    config.num_experts,
                    config.intermediate_size * 2 // self.world_size,
                    config.hidden_size,
                ),
                device=device,
                dtype=torch.bfloat16,
            )
        )
        # 第一层偏置。
        self.mlp1_bias = torch.nn.Parameter(
            torch.empty(
                (config.num_experts, config.intermediate_size * 2 // self.world_size),
                device=device,
                dtype=torch.bfloat16,
            )
        )
        # 第二层专家权重，映射回 hidden_size。
        self.mlp2_weight = torch.nn.Parameter(
            torch.empty(
                (
                    config.num_experts,
                    config.hidden_size,
                    config.intermediate_size // self.world_size,
                ),
                device=device,
                dtype=torch.bfloat16,
            )
        )
        # 第二层偏置。
        self.mlp2_bias = torch.nn.Parameter(
            torch.empty(
                (config.num_experts, config.hidden_size),
                device=device,
                dtype=torch.bfloat16,
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1) 预归一化 + 路由分数。
        t = self.norm(x)
        g = self.gate(t)
        # 2) top-k 专家选择（每个 token 独立选择）。
        experts = torch.topk(g, k=self.experts_per_token, dim=-1, sorted=True)
        # 将 top-k logits 归一化为组合权重。
        expert_weights = torch.nn.functional.softmax(experts.values, dim=1)
        expert_indices = experts.indices

        # MLP #1
        # 按 token 选中的专家索引，批量 gather 对应专家参数。
        mlp1_weight = self.mlp1_weight[expert_indices, ...]
        mlp1_bias = self.mlp1_bias[expert_indices, ...]
        # "beck,bk->bec": 对每个 token/专家执行线性层。
        t = torch.einsum("beck,bk->bec", mlp1_weight, t) + mlp1_bias
        t = swiglu(t, limit=self.swiglu_limit)

        # MLP #2
        mlp2_weight = self.mlp2_weight[expert_indices, ...]
        mlp2_bias = self.mlp2_bias[expert_indices, ...]
        # 第二层把中间表示投回 hidden_size。
        t = torch.einsum("beck,bek->bec", mlp2_weight, t)
        if self.world_size > 1:
            # 张量并行下，各 rank 计算部分和后做 all-reduce 聚合。
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
        t += mlp2_bias

        # Weighted sum of experts
        # 用路由权重对 top-k 专家输出加权求和。
        t = torch.einsum("bec,be->bc", t, expert_weights)

        # 残差连接。
        return x + t


class TransformerBlock(torch.nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        layer_idx: int,
        device: torch.device | None = None,
    ):
        super().__init__()
        # 记录层号，便于与 checkpoint 参数名对齐/调试。
        self.layer_idx = layer_idx
        # 先注意力后 MoE-MLP。
        self.attn = AttentionBlock(config, layer_idx, device)
        self.mlp = MLPBlock(config, device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 顺序执行两个子层，各自内部已包含残差。
        x = self.attn(x)
        x = self.mlp(x)
        return x


class Transformer(torch.nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        device: torch.device | None = None,
    ):
        super().__init__()
        # token id -> hidden 向量。
        self.embedding = torch.nn.Embedding(
            config.vocab_size, config.hidden_size, device=device, dtype=torch.bfloat16
        )
        # 堆叠 N 个 Transformer block。
        self.block = torch.nn.ModuleList(
            [
                TransformerBlock(config, layer_idx, device)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        # 最终归一化 + 词表投影得到 logits。
        self.norm = RMSNorm(config.hidden_size, device=device)
        self.unembedding = torch.nn.Linear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
            device=device,
            dtype=torch.bfloat16,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 输入 x: [T]（token 序列），输出 logits: [T, vocab_size]。
        x = self.embedding(x)
        for block in self.block:
            x = block(x)
        x = self.norm(x)
        x = self.unembedding(x)
        return x

    @staticmethod
    def from_checkpoint(
        path: str, device: str | torch.device = "cuda"
    ) -> "Transformer":
        # 允许字符串设备名（如 "cuda:0"），统一转为 torch.device。
        if not isinstance(device, torch.device):
            device = torch.device(device)

        # 读取模型配置并实例化结构。
        config_path = os.path.join(path, "config.json")
        with open(config_path, "r") as f:
            json_config = json.load(f)
            config = ModelConfig(**json_config)

        model = Transformer(
            config=config,
            device=device,
        )
        model.eval()

        # Load weights
        # 计算当前 rank 的并行切片范围。
        my_rank = dist.get_rank() if dist.is_initialized() else 0
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        per_rank_intermediate_size = config.intermediate_size // world_size

        # Checkpoint 包装器负责普通 tensor 与 MXFP4 权重解码。
        checkpoint = Checkpoint(path, device)

        # 逐参数按名称加载，名称需与 checkpoint 命名规则一致。
        for name, param in model.named_parameters():
            loaded_tensor = checkpoint.get(name)

            # Note: it would be more efficient to do sharding before upcasting from MXFP4,
            # but for simplicity we do it after.
            # mlp1 的 weight/bias 在中间维上按 rank 分片；乘 2 是因为 SwiGLU 双分支。
            if "mlp1" in name:  # both weight and bias
                loaded_tensor = loaded_tensor[
                    :,
                    my_rank * 2
                    * per_rank_intermediate_size : (my_rank + 1) * 2
                    * per_rank_intermediate_size,
                    ...,
                ]
            elif "mlp2_weight" in name:  # only weight
                loaded_tensor = loaded_tensor[
                    ...,
                    my_rank
                    * per_rank_intermediate_size : (my_rank + 1)
                    * per_rank_intermediate_size,
                ]
            try:
                # 使用 inplace copy_ 保持 Parameter 对象与图结构不变。
                param.data.copy_(loaded_tensor)
            except:
                # 加载失败时打印详细形状，便于定位权重命名/切片问题。
                print(f"{name=} {param.data.shape=} {loaded_tensor.shape=}")
                raise

        return model


class TokenGenerator:
    @torch.inference_mode()
    def __init__(self, checkpoint: str, device: torch.device):
        # 推理模式下构建模型，关闭 autograd 以减少显存/开销。
        self.device = device
        self.model = Transformer.from_checkpoint(checkpoint, device=self.device)

    @torch.inference_mode()
    def generate(self,
                 prompt_tokens: list[int],
                 stop_tokens: list[int],
                 temperature: float = 1.0,
                 max_tokens: int = 0,
                 return_logprobs: bool = False):
        # tokens 始终包含 prompt + 已生成 token。
        tokens = list(prompt_tokens)
        num_generated_tokens = 0
        # max_tokens=0 约定为不设上限，直到遇到 stop token。
        while max_tokens == 0 or num_generated_tokens < max_tokens:
            # 每步把当前完整上下文送入模型，取最后一个位置 logits。
            logits = self.model(torch.as_tensor(tokens, dtype=torch.int32, device=self.device))[-1]
            if temperature == 0.0:
                # 贪心解码。
                predicted_token = torch.argmax(logits, dim=-1).item()
            else:
                # 温度采样：先缩放 logits，再按概率多项采样。
                probs = torch.softmax(logits * (1.0 / temperature), dim=-1)
                predicted_token = torch.multinomial(probs, num_samples=1).item()
            tokens.append(predicted_token)
            num_generated_tokens += 1

            if return_logprobs:
                # 返回被采样 token 在当前步的对数概率。
                logprobs = torch.log_softmax(logits, dim=-1)
                selected_logprobs = logprobs[predicted_token].item()
                yield predicted_token, selected_logprobs
            else:
                yield predicted_token

            # 命中任意停止词立即结束生成。
            if predicted_token in stop_tokens:
                break
