class TlamaConfig:
    """
    Configuration class for Transformer Language Model Architecture.
    Holds all hyperparameters for model architecture.
    """
    def __init__(
        self,
        d_model=4096,
        n_layers=32,
        n_heads=32,
        n_kv_heads=None,
        vocab_size=128000,
        multiple_of=256,
        ffn_dim_multiplier=None,
        norm_eps=1e-5,
        rope_theta=500000.0,
        max_batch_size=32,
        max_seq_len=2048,
        use_parallel=False,
        kv_cache=False
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