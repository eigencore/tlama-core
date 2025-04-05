# Copyright 2025 EigenCore.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from tlamacore.models.config import TlamaConfig
from dataclasses import dataclass
from typing import Optional, Tuple
# import fairscale.nn.model_parallel.initialize as fs_init
# from fairscale.nn.model_parallel.layers import (
#     ColumnParallelLinear,
#     RowParallelLinear,
#     VocabParallelEmbedding,
# )

class RMSNorm(torch.nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm)
    
    This normalization is based on the Root Mean Square (RMS) norm instead of mean and variance,
    as in LayerNorm. It is commonly used in language models to improve stability and efficiency.
    
    Attributes:
        eps (float): Small value to prevent division by zero. Default is 1e-6.
        weight (torch.nn.Parameter): Learnable parameter to scale the output.
    
    Parameters:
        dim (int): Input dimension over which normalization is applied.
        eps (float, optional): Value for numerical stability.
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies RMS normalization.
        
        Args:
            x (torch.Tensor): Input tensor.
        
        Returns:
            torch.Tensor: Normalized tensor.
        """
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of RMSNorm.
        
        Args:
            x (torch.Tensor): Input tensor of shape (..., dim).
        
        Returns:
            torch.Tensor: Normalized and scaled tensor.
        """
        output = self._norm(x.float()).type_as(x)
        return output * self.weight
    
def compute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """
    Computes the complex-valued rotary positional embeddings (RoPE) frequencies.

    This function generates a tensor of complex numbers representing the rotary positional embeddings
    used in transformer models. The embeddings are computed in polar form, where the magnitude is 1
    and the angle is determined by the product of position indices and frequency scaling factors.

    Args:
        dim (int): Dimensionality of the embeddings. Typically corresponds to the model's hidden size.
        end (int): Maximum sequence length for which the embeddings are computed.
        theta (float, optional): Scaling factor for the frequencies. Default is 10000.0.

    Returns:
        torch.Tensor: A tensor of shape `(end, dim // 2)` containing complex numbers in polar form.
                      Each complex number has a magnitude of 1 and an angle determined by the
                      position and frequency.
    """
    theta_ = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    m = torch.arange(end, device=theta_.device, dtype=torch.float32)
    freqs = torch.outer(m, theta_)  # m_i * theta_j
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # r*(cos(m_i * theta_j), sin(m_i * theta_j)), r=1
    return freqs_cis

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """
    Reshapes the `freqs_cis` tensor for broadcasting with the input tensor `x`.

    This function adjusts the shape of the `freqs_cis` tensor so that it can be broadcasted
    with the input tensor `x` during operations such as element-wise multiplication. The reshaped
    tensor will have singleton dimensions (`1`) for all axes except the sequence length and embedding
    dimensions, ensuring compatibility with `x`.

    Args:
        freqs_cis (torch.Tensor): A tensor of shape `(seq_len, hidden_dim)` containing complex-valued
                                  rotary positional embeddings.
        x (torch.Tensor): Input tensor of shape `(batch_size, seq_len, hidden_dim)` or similar, where
                          `seq_len` is the sequence length and `hidden_dim` is the embedding size.

    Returns:
        torch.Tensor: A reshaped version of `freqs_cis` with singleton dimensions added, making it
                      compatible for broadcasting with `x`.

    Example:
        >>> freqs_cis = torch.randn(2048, 256)  # (seq_len, hidden_dim)
        >>> x = torch.randn(8, 2048, 256)      # (batch_size, seq_len, hidden_dim)
        >>> reshaped_freqs_cis = reshape_for_broadcast(freqs_cis, x)
        >>> print(reshaped_freqs_cis.shape)
        torch.Size([1, 2048, 256])
    """
    ndim = x.ndim
    assert 0 <= 1 < ndim, "Input tensor `x` must have at least 2 dimensions."
    assert freqs_cis.shape == (x.shape[1], x.shape[-1]), (
        "Shape of `freqs_cis` must match the sequence length and embedding size of `x`."
    )
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)

def apply_rope(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies Rotary Positional Embeddings (RoPE) to the query (`xq`) and key (`xk`) tensors.

    This function incorporates positional information into the query and key tensors by applying
    rotary positional embeddings. The embeddings are applied in the complex domain, where the
    tensors are first converted to complex numbers, multiplied by the positional embeddings, and
    then converted back to real numbers.

    Args:
        xq (torch.Tensor): Query tensor of shape `(batch_size, seq_len, hidden_dim)` containing real values.
        xk (torch.Tensor): Key tensor of shape `(batch_size, seq_len, hidden_dim)` containing real values.
        freqs_cis (torch.Tensor): Complex-valued rotary positional embeddings of shape `(seq_len, hidden_dim // 2)`.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - `xq_out` (torch.Tensor): Query tensor with positional embeddings applied, of shape `(batch_size, seq_len, hidden_dim)`.
            - `xk_out` (torch.Tensor): Key tensor with positional embeddings applied, of shape `(batch_size, seq_len, hidden_dim)`.
    """
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

def repeat_kv(kv: torch.Tensor, n_rep: int) -> torch.Tensor:
    bsz, seq_len, n_kv_heads, head_dim = kv.shape
    if n_rep == 1:
        return kv
    return (
        kv[:, :, :, None, :]
        .expand(bsz, seq_len, n_kv_heads, n_rep, head_dim)
        .reshape(bsz, seq_len, n_kv_heads * n_rep, head_dim)
    )

class Attention(nn.Module):
    """
    Attention mechanism for the Tlama model.

    This class implements the multi-head attention mechanism with rotary positional embeddings (RoPE)
    and key-value caching for efficient transformer computations.

    Attributes:
        n_kv_heads (int): Number of key-value heads for attention.
        n_local_heads (int): Number of local heads for model parallelism.
        n_local_kv_heads (int): Number of local key-value heads for model parallelism.
        n_rep (int): Number of repetitions for key-value heads.
        head_dim (int): Dimensionality of each attention head.
        Wq (ColumnParallelLinear): Linear layer for query projection.
        Wk (ColumnParallelLinear): Linear layer for key projection.
        Wv (ColumnParallelLinear): Linear layer for value projection.
        Wo (RowParallelLinear): Linear layer for output projection.
        cache_k (torch.Tensor): Cache for key tensors.
        cache_v (torch.Tensor): Cache for value tensors.

    Parameters:
        config (TlamaConfig): Configuration object containing model hyperparameters.
    """
    def __init__(self, config: TlamaConfig):
        super().__init__()
        self.n_kv_heads = config.n_kv_heads or config.n_heads 
        model_parallel_size = fs_init.get_model_parallel_world_size() if config.use_parallel else 1
        self.n_local_heads = config.n_heads // model_parallel_size
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = config.d_model // config.n_heads
        
        if model_parallel_size > 1:
            self.Wq = ColumnParallelLinear( # This creates the Wq (query) matrix of shape (d_model, n_heads * head_dim). It is similar to nn.Linear but with model parallelism.
                config.d_model,
                config.n_heads * self.head_dim,
                bias=False,
                gather_output=False,
                init_method=lambda x: x # TODO: Create a smart initializer method
            )
            
            self.Wk = ColumnParallelLinear(
                config.d_model,
                config.n_heads * self.head_dim,
                bias=False,
                gather_output=False,
                init_method=lambda x: x # TODO: Create a smart initializer method
            )
            
            self.Wv = ColumnParallelLinear(
                config.d_model,
                config.n_heads * self.head_dim,
                bias=False,
                gather_output=False,
                init_method=lambda x: x # TODO: Create a smart initializer method
            )
            
            self.Wo = RowParallelLinear(
                config.n_heads * self.head_dim,
                config.d_model,
                input_is_parallel=True,
                bias=False,
                init_method=lambda x: x # TODO: Create a smart initializer method
            )
        else:
            self.Wq = nn.Linear(
                config.d_model,
                config.n_heads * self.head_dim,
                bias=False
            )
            
            self.Wk = nn.Linear(
                config.d_model,
                config.n_heads * self.head_dim,
                bias=False
            )
            
            self.Wv = nn.Linear(
                config.d_model,
                config.n_heads * self.head_dim,
                bias=False
            )
            
            self.Wo = nn.Linear(
                config.n_heads * self.head_dim,
                config.d_model,
                bias=False
            )
        
        if config.kv_cache:
            self.cache_k = torch.zeros(
                (
                    config.max_batch_size,
                    config.max_seq_len,
                    self.n_local_kv_heads,
                    self.head_dim
                )
            ).cuda()
            
            self.cache_v = torch.zeros(
                (
                    config.max_batch_size,
                    config.max_seq_len,
                    self.n_local_kv_heads,
                    self.head_dim
                )
            ).cuda()
        
    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freq_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
        kv_cache: bool = True
    ):
        """
        Forward pass for the attention mechanism.
        
        Args:
            x (torch.Tensor): Input tensor of shape `(batch_size, seq_len, d_model)`.
            start_pos (int): Starting position for the sequence.
            freq_cis (torch.Tensor): Complex-valued rotary positional embeddings.
            mask (Optional[torch.Tensor]): Attention mask tensor.
        Returns:
            torch.Tensor: Output tensor after applying attention.
        """
        
        bsz, seq_len, _ = x.shape
        # Project the embeddings to query, key, and value
        xq, xk, xv = self.Wq(x), self.Wk(x), self.Wv(x)
        
        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        
        xq, xk = apply_rope(xq, xk, freq_cis)
        
        if kv_cache:
            self.cache_k = self.cache_k.to(xq)
            self.cache_v = self.cache_v.to(xq)
            
            self.cache_k[:bsz, start_pos : start_pos + seq_len] = xk
            self.cache_v[:bsz, start_pos : start_pos + seq_len] = xv
            
            keys = self.cache_k[:bsz, : start_pos + seq_len]
            values = self.cache_v[:bsz, : start_pos + seq_len]
            
            # if n_kv_heads < n_heads, we need to replicate the keys and values
            keys = repeat_kv(
                keys,
                self.n_rep
            ) # shape: (bsz, chahe_len + seq_len, n_local_heads, head_dim)
            values = repeat_kv(
                values,
                self.n_rep
            ) # shape: (bsz, cache_len + seq_len, n_local_heads, head_dim)
        else:
            keys = xk
            values = xv
        
        
        xq = xq.transpose(1,2) # (bsz, n_local_heads, seq_len, head_dim)
        keys = keys.transpose(1,2) # (bsz, n_local_heads, cache_len + seq_len, head_dim)
        values = values.transpose(1,2) # (bsz, n_local_heads, cache_len + seq_len, head_dim)
        
        
        # ---------------------------- Manual attention ------------------------------
        
        scores = torch.matmul(xq, keys.transpose(2,3)) / math.sqrt(self.head_dim) 
        
        if mask is not None:
            scores = scores + mask
            
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, values)
        # ----------------------------------------------------------------------------
        
        # NOTE: Needs to evaluate if we can change the manual implementation for F.scaled_dot_product_attention()
        # along with fairscale implementation
        # Use scaled_dot_product_attention
        # output = F.scaled_dot_product_attention(
        #    xq, keys, values,
        #    attn_mask=mask,  # Pass the mask if provided
        #    is_causal=mask is None  # Use causal attention if no mask is provided
        #) 
        
        
        output = output.transpose(1,2).contiguous().view(bsz, seq_len, -1)
        
        return self.Wo(output)
     
class FeedForward(nn.Module):
    """
    FeedForward network for the Tlama model.

    This class implements a feed-forward neural network with optional dimensionality adjustments
    and model parallelism. It uses the SwiGLU activation function for improved performance.

    Attributes:
        w1 (ColumnParallelLinear): First linear layer for the feed-forward network.
        w2 (RowParallelLinear): Second linear layer for the feed-forward network.
        w3 (ColumnParallelLinear): Third linear layer for the feed-forward network.

    Parameters:
        d_model (int): Dimensionality of the model.
        hidden_dim (int): Dimensionality of the hidden layer.
        multiple_of (int): Ensures the hidden layer size is a multiple of this value.
        ffn_dim_multiplier (Optional[float]): Multiplier for feed-forward network dimension.
    """
    def __init__(
        self,
        d_model: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
        use_parallel: bool = True
    ):
        
        super().__init__()
        if ffn_dim_multiplier is not None:
            hidden_dim = int(hidden_dim * ffn_dim_multiplier)
        else:
            hidden_dim = int(2 * hidden_dim / 3)

        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        
        if use_parallel:
            self.w1 = ColumnParallelLinear(
                d_model,
                hidden_dim,
                bias=False,
                gather_output=False,
                init_method=lambda x: x
            )
            
            self.w2 = RowParallelLinear(
                hidden_dim,
                d_model,
                bias=False,
                input_is_parallel=True,
                init_method=lambda x: x
            )
            
            self.w3 = ColumnParallelLinear(
                d_model,
                hidden_dim,
                bias=False,
                gather_output=False,
                init_method=lambda x: x
            )
        else:
            self.w1 = nn.Linear(d_model, hidden_dim, bias=False)
            self.w2 = nn.Linear(hidden_dim, d_model, bias=False)
            self.w3 = nn.Linear(d_model, hidden_dim, bias=False)

        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the feed-forward network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).

        Returns:
            torch.Tensor: Output tensor after applying the feed-forward network.
        """
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
    
class TransformerBlock(nn.Module):
    """
    Transformer block for the Tlama model.

    This class implements a single transformer block, which consists of an attention mechanism
    followed by a feed-forward neural network. Each block also includes layer normalization
    before the attention and feed-forward layers.

    Attributes:
        n_heads (int): Number of attention heads.
        d_model (int): Dimensionality of the model.
        head_dim (int): Dimensionality of each attention head.
        attention (Attention): Attention mechanism for the transformer block.
        ffn (FeedForward): Feed-forward neural network for the transformer block.
        layer_id (int): Identifier for the layer.
        attention_norm (RMSNorm): Layer normalization before the attention mechanism.
        ffn_norm (RMSNorm): Layer normalization before the feed-forward network.

    Parameters:
        layer_id (int): Identifier for the layer.
        config (TlamaConfig): Configuration object containing model hyperparameters.
    """
    def __init__(self, layer_id: int, config: TlamaConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.d_model = config.d_model
        self.head_dim = config.d_model // config.n_heads
        self.attention = Attention(config)
        self.ffn = FeedForward(
            d_model=config.d_model,
            hidden_dim=4*config.d_model,
            multiple_of=config.multiple_of,
            ffn_dim_multiplier=config.ffn_dim_multiplier,
            use_parallel=config.use_parallel
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(config.d_model, eps=config.norm_eps)
        self.ffn_norm = RMSNorm(config.d_model, eps=config.norm_eps)
        
    
    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freq_cis: torch.Tensor,
        mask: Optional[torch.Tensor]
    ):
        """
        Forward pass for the transformer block.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model).
            start_pos (int): Starting position for the sequence.
            freq_cis (torch.Tensor): Complex-valued rotary positional embeddings.
            mask (Optional[torch.Tensor]): Attention mask tensor.

        Returns:
            torch.Tensor: Output tensor after applying the transformer block.
        """
        h = x + self.attention(self.attention_norm(x), start_pos, freq_cis, mask, kv_cache=False)
        out = h + self.ffn(self.ffn_norm(h))
        return out
    
class Transformer(nn.Module):
    """
    Transformer model for the Tlama model.

    This class implements the full transformer model, which consists of an embedding layer,
    multiple transformer blocks, and a final linear layer for output. The model also includes
    rotary positional embeddings (RoPE) and layer normalization.

    Attributes:
        config (TlamaConfig): Configuration object containing model hyperparameters.
        vocab_size (int): Size of the vocabulary.
        n_layers (int): Number of transformer layers.
        token_emb (VocabParallelEmbedding): Embedding layer for input tokens.
        layers (nn.ModuleList): List of transformer blocks.
        norm (RMSNorm): Layer normalization applied to the final output.
        output (ColumnParallelLinear): Linear layer for generating the final output.
        freq_cis (torch.Tensor): Complex-valued rotary positional embeddings.

    Parameters:
        config (TlamaConfig): Configuration object containing model hyperparameters.
    """
    def __init__(self, config: TlamaConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.n_layers = config.n_layers
        self.std = config.weight_init[0]
        self.mean = config.weight_init[1]
         
        self.layers = nn.ModuleList()
        for layer_id in range(self.n_layers):
            self.layers.append(TransformerBlock(layer_id, config))
            
        self.norm = RMSNorm(config.d_model, eps=config.norm_eps)
        
        if config.use_parallel:
            self.output = ColumnParallelLinear(
                config.d_model,
                self.vocab_size,
                bias=False,
                init_method=lambda x: x
            )
            
            
            self.token_emb = VocabParallelEmbedding(
                self.vocab_size,
                config.d_model,
                init_method=lambda x: x
            )
        else:
            
            self.output = nn.Linear(
                config.d_model,
                self.vocab_size,
                bias=False
            )
            
            self.token_emb = nn.Embedding(
                self.vocab_size,
                config.d_model
            )
   
        self.register_buffer(
            "freq_cis",       
            compute_freqs_cis(
                config.d_model // config.n_heads,  # head_dim
                config.max_seq_len * 2,
                config.rope_theta
            )
        )
    
        self.apply(self._init_weights)
        
        
    def forward(
        self,
        tokens: torch.Tensor,
        start_pos: int,
        targets: Optional[torch.Tensor] = None,
    ):
        _bsz, seq_len = tokens.shape
        
        # Debug device placement
        device = tokens.device

        h = self.token_emb(tokens)
        freq_cis = self.freq_cis[start_pos : start_pos + seq_len].to(device)
        
        mask = None
        if seq_len > 1:
            # Create mask on the same device as tokens, with explicit device placement
            try:
                mask = torch.full((seq_len, seq_len), float('-inf'), device=device)
                mask = torch.triu(mask, diagonal=1)
                # Create zeros tensor on the same device
                zeros = torch.zeros((seq_len, start_pos), device=device)
                mask = torch.hstack([zeros, mask]).type_as(h)
            except Exception as e:
                print(f"Error creating mask: {e}")
                # Fallback to CPU if device operations fail, then move to target device
                mask = torch.full((seq_len, seq_len), float('-inf'), device='cpu')
                mask = torch.triu(mask, diagonal=1)
                zeros = torch.zeros((seq_len, start_pos), device='cpu')
                mask = torch.hstack([zeros, mask]).to(device).type_as(h)
        
        for layer in self.layers:
            h = layer(h, start_pos, freq_cis, mask)
        
        h = self.norm(h)
        output = self.output(h).float()
        loss = None
        if targets is not None:
            loss = F.cross_entropy(output.view(-1, output.size(-1)), targets.view(-1))
        return output, loss
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = self.std
            mean = self.mean
            if hasattr(module, 'TLAMA124M_SCALE_INIT'):
                std = (2 * self.config.n_layer) * -0.5
            torch.nn.init.normal_(module.weight, mean=mean, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=mean, std=self.std)
    
    def configure_optimizers(self, weight_decay, learning_rate, device_type, master_process=True):
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8)
        return optimizer
            

        
        
    
    
    