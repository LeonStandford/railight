from __future__ import annotations
import copy
import torch
from torch import nn
from .blocks import RoPEAttention


class MemoryAttentionLayer(nn.Module):

    def __init__(
        self,
        d_model: int = 256,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        pos_enc_at_attn: bool = False,
        pos_enc_at_cross_attn_keys: bool = True,
        pos_enc_at_cross_attn_queries: bool = False,
        self_attn: nn.Module | None = None,
        cross_attn: nn.Module | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.dim_feedforward = dim_feedforward
        self.dropout_value = dropout
        self.self_attn = self_attn or RoPEAttention(
            embedding_dim=256, num_heads=1, downsample_rate=1
        )
        self.cross_attn_image = cross_attn or RoPEAttention(
            rope_k_repeat=True,
            embedding_dim=256,
            num_heads=1,
            downsample_rate=1,
            kv_in_dim=64,
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        self.pos_enc_at_attn = pos_enc_at_attn
        self.pos_enc_at_cross_attn_queries = pos_enc_at_cross_attn_queries
        self.pos_enc_at_cross_attn_keys = pos_enc_at_cross_attn_keys

    def _forward_sa(
        self, tgt: torch.Tensor, query_pos: torch.Tensor | None
    ) -> torch.Tensor:
        tgt2 = self.norm1(tgt)
        q = k = tgt2 + query_pos if self.pos_enc_at_attn else tgt2
        tgt2 = self.self_attn(q, k, v=tgt2)
        tgt = tgt + self.dropout1(tgt2)
        return tgt

    def _forward_ca(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        query_pos: torch.Tensor | None,
        pos: torch.Tensor | None,
        num_k_exclude_rope: int = 0,
    ) -> torch.Tensor:
        kwds = {}
        if num_k_exclude_rope > 0:
            assert isinstance(self.cross_attn_image, RoPEAttention)
            kwds = {"num_k_exclude_rope": num_k_exclude_rope}
        tgt2 = self.norm2(tgt)
        tgt2 = self.cross_attn_image(
            q=tgt2 + query_pos if self.pos_enc_at_cross_attn_queries else tgt2,
            k=memory + pos if self.pos_enc_at_cross_attn_keys else memory,
            v=memory,
            **kwds,
        )
        tgt = tgt + self.dropout2(tgt2)
        return tgt

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        pos: torch.Tensor | None = None,
        query_pos: torch.Tensor | None = None,
        num_k_exclude_rope: int = 0,
    ) -> torch.Tensor:
        tgt = self._forward_sa(tgt, query_pos)
        tgt = self._forward_ca(tgt, memory, query_pos, pos, num_k_exclude_rope)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt


class MemoryAttention(nn.Module):

    def __init__(
        self,
        d_model: int,
        pos_enc_at_input: bool,
        layer: nn.Module,
        num_layers: int,
        batch_first: bool = True,
    ):
        super().__init__()
        self.d_model = d_model
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = nn.LayerNorm(d_model)
        self.pos_enc_at_input = pos_enc_at_input
        self.batch_first = batch_first

    def forward(
        self,
        curr: torch.Tensor,
        memory: torch.Tensor,
        curr_pos: torch.Tensor | None = None,
        memory_pos: torch.Tensor | None = None,
        num_obj_ptr_tokens: int = 0,
    ) -> torch.Tensor:
        if isinstance(curr, list):
            assert isinstance(curr_pos, list)
            assert len(curr) == len(curr_pos) == 1
            curr, curr_pos = (curr[0], curr_pos[0])
        assert (
            curr.shape[1] == memory.shape[1]
        ), "Batch size must be the same for curr and memory"
        output = curr
        if self.pos_enc_at_input and curr_pos is not None:
            output = output + 0.1 * curr_pos
        if self.batch_first:
            output = output.transpose(0, 1)
            curr_pos = curr_pos.transpose(0, 1)
            memory = memory.transpose(0, 1)
            memory_pos = memory_pos.transpose(0, 1)
        for layer in self.layers:
            kwds = {}
            if isinstance(layer.cross_attn_image, RoPEAttention):
                kwds = {"num_k_exclude_rope": num_obj_ptr_tokens}
            output = layer(
                tgt=output, memory=memory, pos=memory_pos, query_pos=curr_pos, **kwds
            )
        normed_output = self.norm(output)
        if self.batch_first:
            normed_output = normed_output.transpose(0, 1)
            curr_pos = curr_pos.transpose(0, 1)
        return normed_output
