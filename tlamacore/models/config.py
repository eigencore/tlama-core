from typing import Optional, Tuple

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
    def __init__(
        self,
        d_model: int = 4096,
        n_layers: int = 32,
        n_heads: int = 32,
        n_kv_heads: Optional[int] = None,
        vocab_size: int = 128000,
        multiple_of: int = 256,
        ffn_dim_multiplier: Optional[float] = None,
        norm_eps: float = 1e-5,
        rope_theta: float = 500000.0,
        max_batch_size: int = 32,
        max_seq_len: int = 2048,
        use_parallel: bool = False,
        kv_cache: bool = False,
        weight_init: Tuple[float, float] = (0.02, 0.0)
    ):
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.vocab_size = vocab_size
        self.multiple_of = multiple_of
        self.ffn_dim_multiplier = ffn_dim_multiplier
        self.norm_eps = norm_eps
        self.rope_theta = rope_theta
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.use_parallel = use_parallel
        self.kv_cache = kv_cache
        self.weight_init = weight_init
        
        



