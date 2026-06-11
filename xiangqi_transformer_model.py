from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass
class XiangqiTransformerConfig:
    in_channels: int = 115
    board_h: int = 10
    board_w: int = 9
    d_model: int = 512
    num_layers: int = 12
    num_heads: int = 8
    ffn_dim: int = 2048
    dropout: float = 0.1
    policy_dim: int = 8100
    policy_head_dim: int = 256
    material_dim: int = 16
    use_2d_relative_attention_bias: bool = False
    use_line_of_sight_attention_bias: bool = False
    use_history_memory_attention: bool = False
    use_global_strategic_attention: bool = False
    use_trunk_global_strategy_tokens: bool = False
    use_value_token_pooling: bool = False
    num_global_strategy_tokens: int = 6
    use_cnn_local_tactical_adapter: bool = False
    cnn_local_channels: int = 128
    cnn_local_blocks: int = 4
    use_cnn_policy_residual_adapter: bool = False
    cnn_policy_channels: int = 128
    cnn_policy_blocks: int = 4
    cnn_policy_rank: int = 64
    use_cnn_local_tactical_stem: bool = False
    cnn_stem_channels: int = 128
    cnn_stem_blocks: int = 4

    def __post_init__(self) -> None:
        num_squares = self.board_h * self.board_w
        expected_policy_dim = num_squares * num_squares
        if self.d_model % self.num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        if self.policy_dim != expected_policy_dim:
            raise ValueError(
                f"policy_dim must equal (board_h * board_w) ** 2, got {self.policy_dim}"
            )
        if self.material_dim != 16:
            raise ValueError("material_dim is fixed at 16 for the planned material token")
        if self.policy_head_dim < 1:
            raise ValueError("policy_head_dim must be >= 1")
        if self.num_global_strategy_tokens < 1:
            raise ValueError("num_global_strategy_tokens must be >= 1")
        if self.cnn_local_channels < 1:
            raise ValueError("cnn_local_channels must be >= 1")
        if self.cnn_local_blocks < 1:
            raise ValueError("cnn_local_blocks must be >= 1")
        if self.cnn_policy_channels < 1:
            raise ValueError("cnn_policy_channels must be >= 1")
        if self.cnn_policy_blocks < 1:
            raise ValueError("cnn_policy_blocks must be >= 1")
        if self.cnn_policy_rank < 1:
            raise ValueError("cnn_policy_rank must be >= 1")
        if self.cnn_stem_channels < 1:
            raise ValueError("cnn_stem_channels must be >= 1")
        if self.cnn_stem_blocks < 1:
            raise ValueError("cnn_stem_blocks must be >= 1")


class CNNLocalTacticalBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        groups = 8 if channels % 8 == 0 else 1
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = self.conv1(F.gelu(self.norm1(x)))
        x = self.conv2(F.gelu(self.norm2(x)))
        return residual + x


class XiangqiTransformerBlock(nn.Module):
    def __init__(self, config: XiangqiTransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.num_squares = config.board_h * config.board_w
        self.prefix_tokens = 1 + (
            int(config.num_global_strategy_tokens)
            if bool(config.use_trunk_global_strategy_tokens) else 0
        )
        self.num_tokens = self.num_squares + self.prefix_tokens
        self.norm1 = nn.LayerNorm(config.d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=config.d_model,
            num_heads=config.num_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.use_2d_relative_attention_bias = bool(config.use_2d_relative_attention_bias)
        if self.use_2d_relative_attention_bias:
            num_relative_positions = (2 * config.board_h - 1) * (2 * config.board_w - 1)
            self.relative_attention_bias = nn.Parameter(
                torch.zeros(config.num_heads, num_relative_positions)
            )
            self.register_buffer(
                "relative_position_index",
                self._build_relative_position_index(config),
                persistent=False,
            )
        self.use_line_of_sight_attention_bias = bool(config.use_line_of_sight_attention_bias)
        if self.use_line_of_sight_attention_bias:
            self.line_of_sight_attention_bias = nn.Parameter(torch.zeros(config.num_heads, 6))
        self.dropout1 = nn.Dropout(config.dropout)

        self.norm2 = nn.LayerNorm(config.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(config.d_model, config.ffn_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.ffn_dim, config.d_model),
            nn.Dropout(config.dropout),
        )

    @staticmethod
    def _build_relative_position_index(config: XiangqiTransformerConfig) -> Tensor:
        num_squares = config.board_h * config.board_w
        board_square_ids = torch.arange(num_squares, dtype=torch.long)
        ranks = board_square_ids // config.board_w
        files = board_square_ids % config.board_w

        rank_delta = ranks[:, None] - ranks[None, :] + (config.board_h - 1)
        file_delta = files[:, None] - files[None, :] + (config.board_w - 1)
        board_index = rank_delta * (2 * config.board_w - 1) + file_delta

        prefix_tokens = 1 + (
            int(config.num_global_strategy_tokens)
            if bool(config.use_trunk_global_strategy_tokens) else 0
        )
        # Prefix tokens (material + optional strategy tokens) have no spatial
        # coordinate, so their interactions keep zero bias. Board-board entries
        # use 2D deltas.
        full_index = torch.full((num_squares + prefix_tokens, num_squares + prefix_tokens), -1, dtype=torch.long)
        full_index[prefix_tokens:, prefix_tokens:] = board_index
        return full_index

    def _relative_attention_bias(self, seq_len: int, device: torch.device) -> Tensor:
        if seq_len != self.num_tokens:
            raise ValueError(f"relative attention bias expected {self.num_tokens} tokens, got {seq_len}")

        index = self.relative_position_index.to(device=device)
        flat_index = index.clamp_min(0).reshape(-1)
        bias = self.relative_attention_bias[:, flat_index]
        bias = bias.reshape(self.config.num_heads, seq_len, seq_len)
        return bias.masked_fill(index.unsqueeze(0) < 0, 0.0).float()

    def forward(
        self,
        x: Tensor,
        line_of_sight_relation: Tensor | None = None,
        line_from_tokens: Tensor | None = None,
        line_to_tokens: Tensor | None = None,
    ) -> Tensor:
        attn_input = self.norm1(x)
        if self.use_2d_relative_attention_bias or self.use_line_of_sight_attention_bias:
            attn_output = self._attention_with_optional_bias(
                attn_input,
                line_of_sight_relation=line_of_sight_relation,
                line_from_tokens=line_from_tokens,
                line_to_tokens=line_to_tokens,
            )
        else:
            attn_output, _ = self.attn(attn_input, attn_input, attn_input, need_weights=False)
        x = x + self.dropout1(attn_output)
        x = x + self.mlp(self.norm2(x))
        return x

    def _attention_with_optional_bias(
        self,
        x: Tensor,
        line_of_sight_relation: Tensor | None,
        line_from_tokens: Tensor | None,
        line_to_tokens: Tensor | None,
    ) -> Tensor:
        batch_size, seq_len, embed_dim = x.shape
        head_dim = embed_dim // self.config.num_heads
        if head_dim * self.config.num_heads != embed_dim:
            raise ValueError("embed_dim must be divisible by num_heads")

        qkv = F.linear(x, self.attn.in_proj_weight, self.attn.in_proj_bias)
        query, key, value = qkv.chunk(3, dim=-1)

        def split_heads(tensor: Tensor) -> Tensor:
            return tensor.view(batch_size, seq_len, self.config.num_heads, head_dim).transpose(1, 2)

        query = split_heads(query)
        key = split_heads(key)
        value = split_heads(value)

        attn_bias = self._attention_bias(
            batch_size=batch_size,
            seq_len=seq_len,
            device=x.device,
            dtype=query.dtype,
            line_of_sight_relation=line_of_sight_relation,
            line_from_tokens=line_from_tokens,
            line_to_tokens=line_to_tokens,
        )
        dropout_p = float(self.attn.dropout) if self.training else 0.0
        try:
            output = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=attn_bias,
                dropout_p=dropout_p,
                is_causal=False,
            )
        except RuntimeError as exc:
            if "not correctly aligned" not in str(exc):
                raise
            output = self._attention_with_scores_fallback(
                query=query,
                key=key,
                value=value,
                attn_bias=attn_bias,
                dropout_p=dropout_p,
            )

        output = output.transpose(1, 2).reshape(batch_size, seq_len, embed_dim)
        return self.attn.out_proj(output)

    def _attention_with_scores_fallback(
        self,
        *,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attn_bias: Tensor | None,
        dropout_p: float,
    ) -> Tensor:
        head_dim = int(query.shape[-1])
        scores = torch.matmul(query.float(), key.float().transpose(-2, -1))
        scores = scores * (head_dim ** -0.5)
        if attn_bias is not None:
            scores = scores + attn_bias.float()
        attention = torch.softmax(scores, dim=-1).to(dtype=value.dtype)
        attention = F.dropout(attention, p=dropout_p, training=self.training)
        return torch.matmul(attention, value)

    def _attention_bias(
        self,
        *,
        batch_size: int,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
        line_of_sight_relation: Tensor | None,
        line_from_tokens: Tensor | None,
        line_to_tokens: Tensor | None,
    ) -> Tensor | None:
        bias: Tensor | None = None
        if self.use_2d_relative_attention_bias:
            bias = self._relative_attention_bias(seq_len, device).to(dtype=dtype).unsqueeze(0)
        if self.use_line_of_sight_attention_bias:
            line_bias = self._line_of_sight_attention_bias(
                batch_size=batch_size,
                seq_len=seq_len,
                device=device,
                dtype=dtype,
                line_of_sight_relation=line_of_sight_relation,
                line_from_tokens=line_from_tokens,
                line_to_tokens=line_to_tokens,
            )
            bias = line_bias if bias is None else bias + line_bias
        return None if bias is None else bias.contiguous()

    def _line_of_sight_attention_bias(
        self,
        *,
        batch_size: int,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
        line_of_sight_relation: Tensor | None,
        line_from_tokens: Tensor | None,
        line_to_tokens: Tensor | None,
    ) -> Tensor:
        if line_of_sight_relation is None or line_from_tokens is None or line_to_tokens is None:
            raise ValueError("line-of-sight attention bias is enabled but pair metadata is missing")
        if line_of_sight_relation.ndim != 2:
            raise ValueError("line_of_sight_relation must have shape [B,P]")

        from_tokens = line_from_tokens.to(device=device)
        to_tokens = line_to_tokens.to(device=device)
        relation = line_of_sight_relation.to(device=device)
        if int(relation.shape[0]) != batch_size:
            raise ValueError(
                "line_of_sight_relation batch dimension mismatch: "
                f"expected {batch_size}, got {int(relation.shape[0])}"
            )

        relation_weight = self.line_of_sight_attention_bias.transpose(0, 1).to(dtype=dtype)
        pair_bias = F.embedding(relation, relation_weight).permute(0, 2, 1).contiguous()

        bias = torch.zeros(
            (batch_size, self.config.num_heads, seq_len, seq_len),
            device=device,
            dtype=dtype,
        )
        bias[:, :, from_tokens, to_tokens] = pair_bias
        return bias


class XiangqiPVTransformer(nn.Module):
    def __init__(self, config: XiangqiTransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.num_squares = config.board_h * config.board_w
        self.policy_head_dim = int(config.policy_head_dim)
        self.num_strategy_tokens = (
            int(config.num_global_strategy_tokens)
            if bool(config.use_trunk_global_strategy_tokens) else 0
        )
        self.board_token_offset = 1 + self.num_strategy_tokens

        self.input_proj = nn.Linear(config.in_channels, config.d_model)
        self.use_cnn_local_tactical_stem = bool(config.use_cnn_local_tactical_stem)
        if self.use_cnn_local_tactical_stem:
            channels = int(config.cnn_stem_channels)
            self.cnn_stem_input = nn.Conv2d(14, channels, kernel_size=1)
            self.cnn_stem_blocks = nn.ModuleList(
                CNNLocalTacticalBlock(channels) for _ in range(int(config.cnn_stem_blocks))
            )
            self.cnn_stem_norm = nn.GroupNorm(8 if channels % 8 == 0 else 1, channels)
            self.cnn_stem_out = nn.Conv2d(channels, config.d_model, kernel_size=1)
        self.use_cnn_local_tactical_adapter = bool(config.use_cnn_local_tactical_adapter)
        if self.use_cnn_local_tactical_adapter:
            channels = int(config.cnn_local_channels)
            self.cnn_local_input = nn.Conv2d(14, channels, kernel_size=1)
            self.cnn_local_blocks = nn.ModuleList(
                CNNLocalTacticalBlock(channels) for _ in range(int(config.cnn_local_blocks))
            )
            self.cnn_local_norm = nn.GroupNorm(8 if channels % 8 == 0 else 1, channels)
            self.cnn_local_out = nn.Conv2d(channels, config.d_model, kernel_size=1)
            nn.init.zeros_(self.cnn_local_out.weight)
            nn.init.zeros_(self.cnn_local_out.bias)
        self.use_cnn_policy_residual_adapter = bool(config.use_cnn_policy_residual_adapter)
        if self.use_cnn_policy_residual_adapter:
            channels = int(config.cnn_policy_channels)
            self.cnn_policy_rank = int(config.cnn_policy_rank)
            self.cnn_policy_input = nn.Conv2d(14, channels, kernel_size=1)
            self.cnn_policy_blocks = nn.ModuleList(
                CNNLocalTacticalBlock(channels) for _ in range(int(config.cnn_policy_blocks))
            )
            self.cnn_policy_norm = nn.GroupNorm(8 if channels % 8 == 0 else 1, channels)
            self.cnn_policy_from = nn.Conv2d(channels, self.cnn_policy_rank, kernel_size=1)
            self.cnn_policy_to = nn.Conv2d(channels, self.cnn_policy_rank, kernel_size=1)
            self.cnn_policy_from_bias = nn.Conv2d(channels, 1, kernel_size=1)
            self.cnn_policy_to_bias = nn.Conv2d(channels, 1, kernel_size=1)
            nn.init.zeros_(self.cnn_policy_to.weight)
            nn.init.zeros_(self.cnn_policy_to.bias)
            nn.init.zeros_(self.cnn_policy_from_bias.weight)
            nn.init.zeros_(self.cnn_policy_from_bias.bias)
            nn.init.zeros_(self.cnn_policy_to_bias.weight)
            nn.init.zeros_(self.cnn_policy_to_bias.bias)
            self.cnn_policy_scale = math.sqrt(float(self.cnn_policy_rank))

        self.square_embedding = nn.Embedding(self.num_squares, config.d_model)
        self.rank_embedding = nn.Embedding(config.board_h, config.d_model)
        self.file_embedding = nn.Embedding(config.board_w, config.d_model)
        self.input_dropout = nn.Dropout(config.dropout)
        self.use_history_memory_attention = bool(config.use_history_memory_attention)
        if self.use_history_memory_attention:
            self.history_memory_proj = nn.Linear(14, config.d_model)
            self.history_memory_frame_embedding = nn.Embedding(7, config.d_model)
            self.history_memory_query_norm = nn.LayerNorm(config.d_model)
            self.history_memory_kv_norm = nn.LayerNorm(config.d_model)
            self.history_memory_attn = nn.MultiheadAttention(
                embed_dim=config.d_model,
                num_heads=config.num_heads,
                dropout=config.dropout,
                batch_first=True,
            )
            self.history_memory_out = nn.Linear(config.d_model, config.d_model)
            nn.init.zeros_(self.history_memory_out.weight)
            nn.init.zeros_(self.history_memory_out.bias)
        self.use_global_strategic_attention = bool(config.use_global_strategic_attention)
        self.global_strategy_feature_dim = 28
        if self.use_global_strategic_attention:
            self.global_strategy_tokens = nn.Parameter(
                torch.empty(config.num_global_strategy_tokens, config.d_model)
            )
            nn.init.normal_(self.global_strategy_tokens, std=0.02)
            self.global_strategy_feature_mlp = nn.Sequential(
                nn.LayerNorm(self.global_strategy_feature_dim),
                nn.Linear(self.global_strategy_feature_dim, config.d_model),
                nn.GELU(),
                nn.Linear(config.d_model, config.d_model),
            )
            self.global_strategy_query_norm = nn.LayerNorm(config.d_model)
            self.global_strategy_board_pool_norm = nn.LayerNorm(config.d_model)
            self.global_strategy_pool_attn = nn.MultiheadAttention(
                embed_dim=config.d_model,
                num_heads=config.num_heads,
                dropout=config.dropout,
                batch_first=True,
            )
            self.global_strategy_board_query_norm = nn.LayerNorm(config.d_model)
            self.global_strategy_token_norm = nn.LayerNorm(config.d_model)
            self.global_strategy_broadcast_attn = nn.MultiheadAttention(
                embed_dim=config.d_model,
                num_heads=config.num_heads,
                dropout=config.dropout,
                batch_first=True,
            )
            self.global_strategy_out = nn.Linear(config.d_model, config.d_model)
            nn.init.zeros_(self.global_strategy_out.weight)
            nn.init.zeros_(self.global_strategy_out.bias)
        self.use_trunk_global_strategy_tokens = bool(config.use_trunk_global_strategy_tokens)
        if self.use_trunk_global_strategy_tokens:
            self.trunk_strategy_tokens = nn.Parameter(
                torch.empty(config.num_global_strategy_tokens, config.d_model)
            )
            nn.init.normal_(self.trunk_strategy_tokens, std=0.02)
            self.trunk_strategy_feature_mlp = nn.Sequential(
                nn.LayerNorm(self.global_strategy_feature_dim),
                nn.Linear(self.global_strategy_feature_dim, config.d_model),
                nn.GELU(),
                nn.Linear(config.d_model, config.d_model),
            )

        self.material_mlp = nn.Sequential(
            nn.Linear(config.material_dim, 256),
            nn.GELU(),
            nn.Linear(256, config.d_model),
        )

        self.blocks = nn.ModuleList(
            XiangqiTransformerBlock(config) for _ in range(config.num_layers)
        )
        self.final_norm = nn.LayerNorm(config.d_model)

        self.from_repr = nn.Linear(config.d_model, self.policy_head_dim)
        self.to_repr = nn.Linear(config.d_model, self.policy_head_dim)
        self.from_bias = nn.Linear(config.d_model, 1)
        self.to_bias = nn.Linear(config.d_model, 1)

        self.value_shared = nn.Sequential(
            nn.LayerNorm(config.d_model),
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        self.use_value_token_pooling = bool(config.use_value_token_pooling)
        if self.use_value_token_pooling:
            self.value_query = nn.Parameter(torch.empty(1, config.d_model))
            nn.init.normal_(self.value_query, std=0.02)
            self.value_pool_norm = nn.LayerNorm(config.d_model)
            self.value_pool_attn = nn.MultiheadAttention(
                embed_dim=config.d_model,
                num_heads=config.num_heads,
                dropout=config.dropout,
                batch_first=True,
            )
        self.wdl_head = nn.Linear(config.d_model, 3)
        self.scalar_head = nn.Linear(config.d_model, 1)

        square_ids = torch.arange(self.num_squares, dtype=torch.long)
        rank_ids = square_ids // config.board_w
        file_ids = square_ids % config.board_w
        self.register_buffer("square_ids", square_ids, persistent=False)
        self.register_buffer("rank_ids", rank_ids, persistent=False)
        self.register_buffer("file_ids", file_ids, persistent=False)
        line_from, line_to, line_relation, line_between = self._build_line_of_sight_buffers(config)
        self.register_buffer("line_from_tokens", line_from + self.board_token_offset, persistent=False)
        self.register_buffer("line_to_tokens", line_to + self.board_token_offset, persistent=False)
        self.register_buffer("line_relation_base", line_relation, persistent=False)
        self.register_buffer("line_between_mask", line_between, persistent=False)

        max_piece_counts = torch.tensor([1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 5.0], dtype=torch.float32)
        piece_values = torch.tensor([0.0, 2.0, 2.0, 4.0, 9.0, 4.5, 1.0], dtype=torch.float32)
        self.register_buffer("max_piece_counts", max_piece_counts, persistent=False)
        self.register_buffer("piece_values", piece_values, persistent=False)
        self.policy_scale = math.sqrt(float(self.policy_head_dim))

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        self._validate_input(x)

        board_tokens = self._tokenize_board(x)
        material_token = self._build_material_token(x).unsqueeze(1)
        prefix_tokens = [material_token]
        if self.use_trunk_global_strategy_tokens:
            prefix_tokens.append(self._trunk_strategy_tokens(x))
        sequence = torch.cat([*prefix_tokens, board_tokens], dim=1)
        sequence = self.input_dropout(sequence)
        line_of_sight_relation = (
            self._line_of_sight_relation(x)
            if self.config.use_line_of_sight_attention_bias else None
        )

        for block in self.blocks:
            sequence = block(
                sequence,
                line_of_sight_relation=line_of_sight_relation,
                line_from_tokens=self.line_from_tokens,
                line_to_tokens=self.line_to_tokens,
            )
        sequence = self.final_norm(sequence)

        material_hidden = sequence[:, 0, :]
        strategy_hidden = (
            sequence[:, 1:self.board_token_offset, :]
            if self.use_trunk_global_strategy_tokens else None
        )
        board_hidden = sequence[:, self.board_token_offset:, :]

        policy_logits = self._compute_policy_logits(board_hidden)
        if self.use_cnn_policy_residual_adapter:
            policy_logits = policy_logits + self._cnn_policy_residual_logits(x)

        value_input = self._value_pool(sequence, material_hidden, strategy_hidden, board_hidden)
        value_hidden = self.value_shared(value_input)
        wdl_logits = self.wdl_head(value_hidden)
        value_scalar = torch.tanh(self.scalar_head(value_hidden))

        return {
            "policy_logits": policy_logits,
            "wdl_logits": wdl_logits,
            "value_scalar": value_scalar,
        }

    def _trunk_strategy_tokens(self, x: Tensor) -> Tensor:
        batch_size = int(x.shape[0])
        strategy_features = self._extract_global_strategy_features(x)
        feature_context = self.trunk_strategy_feature_mlp(strategy_features).unsqueeze(1)
        tokens = self.trunk_strategy_tokens.to(device=x.device, dtype=feature_context.dtype).unsqueeze(0)
        return tokens.expand(batch_size, -1, -1) + feature_context

    @staticmethod
    def _build_line_of_sight_buffers(config: XiangqiTransformerConfig) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        pair_from: list[int] = []
        pair_to: list[int] = []
        relation_base: list[int] = []
        between_masks: list[Tensor] = []
        num_squares = config.board_h * config.board_w
        for src in range(num_squares):
            src_rank, src_file = divmod(src, config.board_w)
            for dst in range(num_squares):
                if src == dst:
                    continue
                dst_rank, dst_file = divmod(dst, config.board_w)
                mask = torch.zeros(num_squares, dtype=torch.float32)
                if src_rank == dst_rank:
                    lo, hi = sorted((src_file, dst_file))
                    for file_id in range(lo + 1, hi):
                        mask[src_rank * config.board_w + file_id] = 1.0
                    base = 0
                elif src_file == dst_file:
                    lo, hi = sorted((src_rank, dst_rank))
                    for rank_id in range(lo + 1, hi):
                        mask[rank_id * config.board_w + src_file] = 1.0
                    base = 3
                else:
                    continue
                pair_from.append(src)
                pair_to.append(dst)
                relation_base.append(base)
                between_masks.append(mask)

        return (
            torch.tensor(pair_from, dtype=torch.long),
            torch.tensor(pair_to, dtype=torch.long),
            torch.tensor(relation_base, dtype=torch.long),
            torch.stack(between_masks, dim=0),
        )

    def _line_of_sight_relation(self, x: Tensor) -> Tensor:
        batch_size = int(x.shape[0])

        current_piece_planes = x[:, :14, :, :]
        occupancy = current_piece_planes.sum(dim=1).reshape(batch_size, self.num_squares)
        occupancy = (occupancy > 0.5).float()
        between_mask = self.line_between_mask.to(device=x.device, dtype=occupancy.dtype)
        blocker_counts = torch.matmul(occupancy, between_mask.transpose(0, 1)).long().clamp_max(2)
        return self.line_relation_base.to(device=x.device).unsqueeze(0) + blocker_counts

    def _validate_input(self, x: Tensor) -> None:
        if not isinstance(x, torch.Tensor):
            raise TypeError("forward expects a torch.Tensor input")
        expected_shape = (
            self.config.in_channels,
            self.config.board_h,
            self.config.board_w,
        )
        if x.ndim != 4:
            raise ValueError(
                f"input must have shape [B,{expected_shape[0]},{expected_shape[1]},{expected_shape[2]}], "
                f"got ndim={x.ndim}"
            )
        if tuple(x.shape[1:]) != expected_shape:
            raise ValueError(
                f"input must have shape [B,{expected_shape[0]},{expected_shape[1]},{expected_shape[2]}], "
                f"got {tuple(x.shape)}"
            )
        if x.dtype != torch.float32:
            raise TypeError(f"input must be float32, got {x.dtype}")

    def _tokenize_board(self, x: Tensor) -> Tensor:
        board = x.permute(0, 2, 3, 1).reshape(x.shape[0], self.num_squares, self.config.in_channels)
        board = self.input_proj(board)
        if self.use_cnn_local_tactical_stem:
            board = board + self._cnn_local_tactical_stem_context(x)
        board = board + self._position_encoding(board.device)
        if self.use_cnn_local_tactical_adapter:
            board = board + self._cnn_local_tactical_context(x)
        if self.use_global_strategic_attention:
            board = board + self._global_strategy_context(board, x)
        if self.use_history_memory_attention:
            board = board + self._history_memory_context(board, x)
        return board

    def _cnn_local_tactical_stem_context(self, x: Tensor) -> Tensor:
        current_piece_planes = x[:, :14, :, :]
        local = self.cnn_stem_input(current_piece_planes)
        for block in self.cnn_stem_blocks:
            local = block(local)
        local = self.cnn_stem_out(F.gelu(self.cnn_stem_norm(local)))
        return local.permute(0, 2, 3, 1).reshape(x.shape[0], self.num_squares, self.config.d_model)

    def _cnn_local_tactical_context(self, x: Tensor) -> Tensor:
        current_piece_planes = x[:, :14, :, :]
        local = self.cnn_local_input(current_piece_planes)
        for block in self.cnn_local_blocks:
            local = block(local)
        local = self.cnn_local_out(F.gelu(self.cnn_local_norm(local)))
        return local.permute(0, 2, 3, 1).reshape(x.shape[0], self.num_squares, self.config.d_model)

    def _cnn_policy_residual_logits(self, x: Tensor) -> Tensor:
        current_piece_planes = x[:, :14, :, :]
        local = self.cnn_policy_input(current_piece_planes)
        for block in self.cnn_policy_blocks:
            local = block(local)
        local = F.gelu(self.cnn_policy_norm(local))

        from_repr = self.cnn_policy_from(local)
        to_repr = self.cnn_policy_to(local)
        from_bias = self.cnn_policy_from_bias(local)
        to_bias = self.cnn_policy_to_bias(local)

        from_repr = from_repr.permute(0, 2, 3, 1).reshape(x.shape[0], self.num_squares, self.cnn_policy_rank)
        to_repr = to_repr.permute(0, 2, 3, 1).reshape(x.shape[0], self.num_squares, self.cnn_policy_rank)
        from_bias = from_bias.permute(0, 2, 3, 1).reshape(x.shape[0], self.num_squares, 1)
        to_bias = to_bias.permute(0, 2, 3, 1).reshape(x.shape[0], self.num_squares, 1).transpose(1, 2)

        logits = torch.matmul(from_repr, to_repr.transpose(1, 2)) / self.cnn_policy_scale
        logits = logits + from_bias + to_bias
        return logits.reshape(x.shape[0], self.config.policy_dim)

    def _global_strategy_context(self, board: Tensor, x: Tensor) -> Tensor:
        batch_size = int(x.shape[0])
        strategy_features = self._extract_global_strategy_features(x)
        feature_context = self.global_strategy_feature_mlp(strategy_features).unsqueeze(1)
        strategy_query = (
            self.global_strategy_tokens.to(device=x.device, dtype=board.dtype).unsqueeze(0)
            + feature_context
        )
        strategy_query = self.global_strategy_query_norm(strategy_query)
        board_key_value = self.global_strategy_board_pool_norm(board)
        strategy_tokens, _ = self.global_strategy_pool_attn(
            strategy_query,
            board_key_value,
            board_key_value,
            need_weights=False,
        )

        board_query = self.global_strategy_board_query_norm(board)
        strategy_key_value = self.global_strategy_token_norm(strategy_tokens)
        context, _ = self.global_strategy_broadcast_attn(
            board_query,
            strategy_key_value,
            strategy_key_value,
            need_weights=False,
        )
        return self.global_strategy_out(context).reshape(batch_size, self.num_squares, self.config.d_model)

    def _value_pool(
        self,
        sequence: Tensor,
        material_hidden: Tensor,
        strategy_hidden: Tensor | None,
        board_hidden: Tensor,
    ) -> Tensor:
        if not self.use_value_token_pooling:
            return material_hidden
        batch_size = int(sequence.shape[0])
        query = self.value_query.to(device=sequence.device, dtype=sequence.dtype).unsqueeze(0)
        query = query.expand(batch_size, -1, -1)
        key_value = self.value_pool_norm(sequence)
        pooled, _ = self.value_pool_attn(query, key_value, key_value, need_weights=False)
        return pooled.squeeze(1)

    def _extract_global_strategy_features(self, x: Tensor) -> Tensor:
        batch_size = int(x.shape[0])
        current_frame = x[:, :14, :, :]
        material_features = self._extract_material_features(x)

        occupancy = current_frame.sum(dim=1).reshape(batch_size, self.num_squares)
        occupancy = (occupancy > 0.5).float()
        relation = self._line_of_sight_relation(x)
        line_hist = F.one_hot(relation, num_classes=6).float().mean(dim=1).to(dtype=x.dtype)

        pieces_by_square = current_frame.reshape(batch_size, 14, self.num_squares).transpose(1, 2)
        from_squares = (self.line_from_tokens - self.board_token_offset).to(device=x.device)
        to_squares = (self.line_to_tokens - self.board_token_offset).to(device=x.device)
        from_piece = pieces_by_square[:, from_squares, :]
        to_piece = pieces_by_square[:, to_squares, :]
        one_screen = relation.eq(1) | relation.eq(4)
        self_cannon_source = from_piece[:, :, 5] > 0.5
        opp_cannon_source = from_piece[:, :, 12] > 0.5
        target_opp = to_piece[:, :, 7:14].sum(dim=2) > 0.5
        target_self = to_piece[:, :, :7].sum(dim=2) > 0.5
        self_cannon_targets = (
            (one_screen & self_cannon_source & target_opp).float().sum(dim=1, keepdim=True) / 16.0
        ).to(dtype=x.dtype)
        opp_cannon_targets = (
            (one_screen & opp_cannon_source & target_self).float().sum(dim=1, keepdim=True) / 16.0
        ).to(dtype=x.dtype)
        king_file_features = self._king_file_exposure_features(current_frame, occupancy).to(dtype=x.dtype)

        return torch.cat(
            [
                material_features,
                line_hist,
                self_cannon_targets,
                opp_cannon_targets,
                king_file_features,
            ],
            dim=1,
        )

    def _king_file_exposure_features(self, current_frame: Tensor, occupancy: Tensor) -> Tensor:
        batch_size = int(current_frame.shape[0])
        device = current_frame.device
        pieces_by_plane = current_frame.reshape(batch_size, 14, self.num_squares)
        self_king = pieces_by_plane[:, 0, :]
        opp_king = pieces_by_plane[:, 7, :]
        self_present = self_king.sum(dim=1) > 0.5
        opp_present = opp_king.sum(dim=1) > 0.5
        both_present = self_present & opp_present

        self_sq = self_king.argmax(dim=1)
        opp_sq = opp_king.argmax(dim=1)
        rank_ids = self.rank_ids.to(device=device)
        file_ids = self.file_ids.to(device=device)
        self_rank = rank_ids[self_sq]
        opp_rank = rank_ids[opp_sq]
        self_file = file_ids[self_sq]
        opp_file = file_ids[opp_sq]

        same_file = (self_file == opp_file) & both_present
        lo_rank = torch.minimum(self_rank, opp_rank)
        hi_rank = torch.maximum(self_rank, opp_rank)
        between = (
            (file_ids.unsqueeze(0) == self_file.unsqueeze(1))
            & (rank_ids.unsqueeze(0) > lo_rank.unsqueeze(1))
            & (rank_ids.unsqueeze(0) < hi_rank.unsqueeze(1))
        )
        blockers = (occupancy * between.float()).sum(dim=1)
        clamped = blockers.clamp_max(2.0)
        same_file_f = same_file.float().unsqueeze(1)
        no_blocker = (same_file & blockers.eq(0)).float().unsqueeze(1)
        one_blocker = (same_file & blockers.eq(1)).float().unsqueeze(1)
        blocker_norm = (clamped / 2.0).unsqueeze(1) * same_file_f
        return torch.cat([same_file_f, no_blocker, one_blocker, blocker_norm], dim=1)

    def _history_memory_context(self, board: Tensor, x: Tensor) -> Tensor:
        if self.config.in_channels < 112:
            raise ValueError("history memory attention requires 8-frame 115-plane input")

        batch_size = int(x.shape[0])
        history = x[:, 14:112, :, :].reshape(
            batch_size,
            7,
            14,
            self.config.board_h,
            self.config.board_w,
        )
        memory = history.permute(0, 1, 3, 4, 2).reshape(batch_size, 7 * self.num_squares, 14)
        memory = self.history_memory_proj(memory)

        frame_ids = torch.arange(7, device=x.device).repeat_interleave(self.num_squares)
        square_ids = self.square_ids.to(x.device).repeat(7)
        rank_ids = self.rank_ids.to(x.device).repeat(7)
        file_ids = self.file_ids.to(x.device).repeat(7)
        memory = (
            memory
            + self.history_memory_frame_embedding(frame_ids).unsqueeze(0)
            + self.square_embedding(square_ids).unsqueeze(0)
            + self.rank_embedding(rank_ids).unsqueeze(0)
            + self.file_embedding(file_ids).unsqueeze(0)
        )

        query = self.history_memory_query_norm(board)
        key_value = self.history_memory_kv_norm(memory)
        context, _ = self.history_memory_attn(query, key_value, key_value, need_weights=False)
        return self.history_memory_out(context)

    def _position_encoding(self, device: torch.device) -> Tensor:
        square_pos = self.square_embedding(self.square_ids.to(device))
        rank_pos = self.rank_embedding(self.rank_ids.to(device))
        file_pos = self.file_embedding(self.file_ids.to(device))
        return (square_pos + rank_pos + file_pos).unsqueeze(0)

    def _build_material_token(self, x: Tensor) -> Tensor:
        material_features = self._extract_material_features(x)
        return self.material_mlp(material_features)

    def _extract_material_features(self, x: Tensor) -> Tensor:
        current_frame = x[:, :14, :, :]
        piece_counts = current_frame.sum(dim=(2, 3))

        self_counts = piece_counts[:, :7]
        opp_counts = piece_counts[:, 7:14]

        normalized_self = self_counts / self.max_piece_counts.to(device=x.device, dtype=x.dtype)
        normalized_opp = opp_counts / self.max_piece_counts.to(device=x.device, dtype=x.dtype)

        piece_values = self.piece_values.to(device=x.device, dtype=x.dtype)
        self_value = (self_counts * piece_values).sum(dim=1, keepdim=True)
        opp_value = (opp_counts * piece_values).sum(dim=1, keepdim=True)
        material_balance = (self_value - opp_value) / 48.0
        phase = (self_value + opp_value) / 96.0

        return torch.cat([normalized_self, normalized_opp, material_balance, phase], dim=1)

    def _compute_policy_logits(self, board_hidden: Tensor) -> Tensor:
        from_repr = self.from_repr(board_hidden)
        to_repr = self.to_repr(board_hidden)
        from_bias = self.from_bias(board_hidden)
        to_bias = self.to_bias(board_hidden).transpose(1, 2)

        logits = torch.matmul(from_repr, to_repr.transpose(1, 2)) / self.policy_scale
        logits = logits + from_bias + to_bias
        return logits.reshape(board_hidden.shape[0], self.config.policy_dim)


def coerce_xiangqi_transformer_config(raw_config: object | None) -> XiangqiTransformerConfig:
    if raw_config is None:
        return XiangqiTransformerConfig()
    if isinstance(raw_config, XiangqiTransformerConfig):
        return raw_config
    if isinstance(raw_config, dict):
        allowed = XiangqiTransformerConfig.__dataclass_fields__.keys()
        return XiangqiTransformerConfig(**{k: v for k, v in raw_config.items() if k in allowed})
    raise TypeError(f"unsupported model_config type: {type(raw_config).__name__}")


def config_from_checkpoint_state(state: dict) -> XiangqiTransformerConfig:
    return coerce_xiangqi_transformer_config(state.get("model_config") or state.get("config"))


def normalize_model_state_dict_keys(state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
    keys = list(state_dict.keys())
    for prefix in ("module.", "_orig_mod.", "model."):
        if keys and all(k.startswith(prefix) for k in keys):
            return {k[len(prefix):]: v for k, v in state_dict.items()}
    return state_dict


def load_xiangqi_model_state_dict(
    model: nn.Module,
    state_dict: dict[str, Tensor],
    *,
    allow_missing_relative_bias: bool = True,
) -> None:
    normalized = normalize_model_state_dict_keys(state_dict)
    incompatible = model.load_state_dict(normalized, strict=False)
    missing = list(incompatible.missing_keys)
    unexpected = list(incompatible.unexpected_keys)
    if allow_missing_relative_bias:
        missing = [
            key for key in missing
            if not (
                key.endswith("relative_attention_bias")
                or key.endswith("line_of_sight_attention_bias")
                or key.startswith("history_memory_")
                or key.startswith("global_strategy_")
                or key.startswith("trunk_strategy_")
                or key.startswith("cnn_local_")
                or key.startswith("cnn_policy_")
                or key.startswith("cnn_stem_")
                or key == "value_query"
                or key.startswith("value_pool_")
            )
        ]
    if missing or unexpected:
        raise RuntimeError(
            "checkpoint/model mismatch: "
            f"missing_keys={missing}, unexpected_keys={unexpected}"
        )


def build_model_from_checkpoint_state(state: dict) -> XiangqiPVTransformer:
    model = XiangqiPVTransformer(config_from_checkpoint_state(state))
    load_xiangqi_model_state_dict(model, state["model_state_dict"])
    return model


__all__ = [
    "XiangqiTransformerConfig",
    "XiangqiPVTransformer",
    "build_model_from_checkpoint_state",
    "coerce_xiangqi_transformer_config",
    "config_from_checkpoint_state",
    "load_xiangqi_model_state_dict",
    "normalize_model_state_dict_keys",
]
