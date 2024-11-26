import torch
from torch import nn
from typing import Optional, Tuple, Union

"""
1D-convolutional layer as defined by Radford et al. for OpenAI GPT (and also used in GPT-2).

Basically works like a linear layer but the weights are transposed.

Args:
    nf (`int`): The number of output features.
    nx (`int`): The number of input features.
"""
class Conv1D(nn.Module):
    def __init__(self, nf, nx):
        super().__init__()
        self.nf = nf
        self.weight = nn.Parameter(torch.empty(nx, nf))
        self.bias = nn.Parameter(torch.zeros(nf))
        nn.init.normal_(self.weight, std=0.02)

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        x = x.view(size_out)
        return x

class Attention(nn.Module):
    def __init__(self, config, is_cross_attention=False, layer_idx=None):
        super().__init__()
        self.config = config
        max_positions = config.max_position_embeddings
        self.register_buffer(
            "bias",
            torch.tril(torch.ones((max_positions, max_positions), dtype=torch.bool)).view(
                1, 1, max_positions, max_positions
            ),
            persistent=False,
        )
        self.register_buffer('masked_bias', torch.tensor(-1e4), persistent=False)

        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.split_size = self.embed_dim
        if self.num_heads * self.head_dim != self.embed_dim:
            raise ValueError(
                f"`embed_dim` must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads}). "
            )
        
        self.scale_attn_weights = config.scale_attn_weights

        # Layer-wise attention scaling, reordering, and upcasting
        self.scale_attn_by_inverse_layer_idx = config.scale_attn_by_inverse_layer_idx
        self.layer_idx = layer_idx
        self.reorder_and_upcast_attn = config.reorder_and_upcast_attn

        #GPT2方式的实现，普通实现可以用Linear层替换
        self.c_attn = Conv1D(3 * self.embed_dim, self.embed_dim)
        self.c_proj = Conv1D(self.embed_dim, self.embed_dim)

        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        self.is_causal = True

        self.pruned_heads = set()

        def _attn(self, query, key, value, attention_mask=None, head_mask=None):
            attn_weights = torch.matmul(query, key.transpose(-1, -2))

            if self.scale_attn_weights:
                attn_weights = attn_weights / torch.full(
                    [], value.size(-1) ** 0.5, dtype=attn_weights.dtype, device=attn_weights.device
                )

            # Layer-wise attention scaling
            if self.scale_attn_by_inverse_layer_idx:
                attn_weights = attn_weights / float(self.layer_idx + 1)

            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask

            attn_weights = nn.functional.softmax(attn_weights, dim=-1)

            # Downcast (if necessary) back to V's dtype (if in mixed-precision) -- No-Op otherwise
            attn_weights = attn_weights.type(value.dtype)
            attn_weights = self.attn_dropout(attn_weights)

            # Mask heads if we want to
            if head_mask is not None:
                attn_weights = attn_weights * head_mask

            attn_output = torch.matmul(attn_weights, value)

            return attn_output, attn_weights

            

        def _split_heads(self, tensor, num_heads, attn_head_size):
            new_shape = tensor.size()[:-1] + (num_heads, attn_head_size)
            tensor = tensor.view(new_shape)
            return tensor.permute(0, 2, 1, 3)
        
        def _merge_heads(self, tensor, num_heads, attn_head_size):
            tensor = tensor.permute(0, 2, 1, 3).contiguous()
            new_shape = tensor.size()[:-2] + (num_heads * attn_head_size)
            return tensor.view(new_shape)

        def forward(
            self,
            hidden_states: Optional[Tuple[torch.FloatTensor]],
            layer_past: Optional[Tuple[torch.Tensor]] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = False,
            output_attentions: Optional[bool] = False
        )  -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]], ...]:
            bsz, _, _ = hidden_states.size()
            query, key, value = self.c_attn(hidden_states).split(self.split_size, dim=2)

            query = self._split_heads(query, self.num_heads, self.head_dim)
            key = self._split_heads(key, self.num_heads, self.head_dim)
            value = self._split_heads(value, self.num_heads, self.head_dim)

            if layer_past is not None:
                past_key, past_value = layer_past
                key = torch.cat((past_key, key), dim=-2)
                value = torch.cat((past_value, value), dim=-2)

            if use_cache is True:
                present = (key, value)
            else:
                present = None

            attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)

            attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
            attn_output = self.c_proj(attn_output)
            attn_output = self.resid_dropout(attn_output)

            outputs = (attn_output, present)
            if output_attentions:
                outputs += (attn_weights,)

            return outputs