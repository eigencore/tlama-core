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


import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional


@dataclass
class TlamaConfig:
    """
    Configuration class for the Tlama model.
    
    Defines model hyperparameters such as the number of layers, dimensions, and other key settings.
    
    Attributes:
        d_model (int): Dimensionality of the model. Default is 4096.
        n_layers (int): Number of transformer layers. Default is 32.
        n_kv_heads (Optional[int]): Number of key-value heads for attention. Default is None (follows n_heads).
        vocab_size (int): Size of the vocabulary. Default is -1 (must be set manually).
        multiple_of (int): Ensures the hidden layer size in SwiGLU is a multiple of this value. Default is 256.
        ffn_dim_multiplier (Optional[float]): Multiplier for feed-forward network dimension. Default is None.
        norm_eps (float): Epsilon value for normalization layers. Default is 1e-5.
        rope_theta (float): Theta value for RoPE positional embeddings. Default is 500000.
        max_batch_size (int): Maximum batch size. Default is 32.
        max_seq_len (int): Maximum sequence length. Default is 2048.
    """
    d_model: int = 4096
    n_layers: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1
    multiple_of: int = 256  # Ensures hidden layer size in SwiGLU is a multiple of this value
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    rope_theta: float = 500000
    
    max_batch_size: int = 32
    max_seq_len: int = 2048

    


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
