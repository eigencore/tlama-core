# Copyright 2025-present Max Galindo & the EigenCore team. All rights reserved.
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

class RotaryPositionEncoding(torch.nn.Module):
    
    """
    This class implements the rotary position encoding as described in the paper https://arxiv.org/pdf/2104.09864.
    
    params:
    - d_model: dimension of the model (embedding dimension)
    - max_len: maximum length of the sequence
    - base: base of the sinusoidal function
    - device: device to run the model on
    - config: special configuration
    """
    
    def __init__(self, d_model, max_len, base=10000, device='cuda', config=None):
    
        super().__init__()
        if config is not None:
            base = config.rope_theta
            partial_rotary_factor = config.partial_rotary_factor if hasattr(config, 'partial_rotary_factor') else 1.0
            d_model = getattr(config, 'head_dim', None)
            if d_model is None: d_model = int((config.hidden_size // config.num_attention_heads))
            device = "cuda"
            max_len = config.max_position_embeddings
        
        self.d_model = d_model
        self.max_len = max_len
        self.base = base
        # We make a dynamic RoTE. We first set it to a max of 4 * 8192 tokens then we iteratively grow it.
        self.current_rope_size = min(4 * 8192, self.max_len)
        
        # In order to make `torch.jit.trace` work, we need to initialize the tensor with a fixed size.
        self._set_cos_sin_cache(seq_len=self.current_rope_size, device=device, dtype=torch.get_default_dtype())
    
    def _set_cos_sin_cache(self, seq_len, device, dtype):
        pass
                    