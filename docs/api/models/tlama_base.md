# Transformer Architecture

The `tlama-core` library implements a transformer-based language model architecture with rotary positional embeddings, RMSNorm layer normalization, and optional model parallelism support. This document explains each component of the architecture and provides examples of how to use them.

## Core Components

- [`RMSNorm`](#rmsnorm): Root Mean Square Layer Normalization
- [`Attention`](#attention-mechanism): Multi-head attention with rotary embeddings
- [`FeedForward`](#feedforward-network): SwiGLU-based feed-forward network
- [`TransformerBlock`](#transformer-block): Combined attention and feed-forward layers
- [`Transformer`](#transformer-model): Complete transformer architecture

## Utility Functions

- [`compute_freqs_cis`](#rotary-positional-embeddings): Generates rotary positional embeddings
- [`apply_rope`](#rotary-positional-embeddings): Applies rotary embeddings to query and key tensors
- [`repeat_kv`](#key-value-repetition): Repeats key-value heads for grouped-query attention

---

## RMSNorm

Root Mean Square Layer Normalization provides a computationally efficient alternative to traditional LayerNorm by using root mean square normalization.

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `dim` | int | Input dimension over which normalization is applied |
| `eps` | float, optional | Small value for numerical stability (default: 1e-6) |

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `weight` | Parameter | Learnable scaling parameter |
| `eps` | float | Epsilon value for numerical stability |

### Methods

#### `_norm(x)`

Applies the RMS normalization operation.

| Parameter | Type | Description |
|-----------|------|-------------|
| `x` | torch.Tensor | Input tensor |

**Returns**: Normalized tensor (without scaling).

#### `forward(x)`

Applies full RMSNorm with learnable scaling.

| Parameter | Type | Description |
|-----------|------|-------------|
| `x` | torch.Tensor | Input tensor of shape (..., dim) |

**Returns**: Normalized and scaled tensor.

### Example

```python
import torch
from tlama_core.models import RMSNorm

# Create RMSNorm layer
norm_layer = RMSNorm(dim=512, eps=1e-5)

# Apply normalization to an input tensor
x = torch.randn(32, 128, 512)  # (batch_size, seq_len, hidden_dim)
normalized_x = norm_layer(x)
```

---

## Rotary Positional Embeddings

Rotary Positional Embeddings (RoPE) encode position information directly into attention mechanisms, allowing for better relative position understanding.

### Functions

#### `compute_freqs_cis(dim, end, theta=10000.0)`

Computes the complex-valued rotary positional embeddings frequencies.

| Parameter | Type | Description |
|-----------|------|-------------|
| `dim` | int | Dimensionality of the embeddings (head dimension) |
| `end` | int | Maximum sequence length for which to compute embeddings |
| `theta` | float, optional | Scaling factor for the frequencies (default: 10000.0) |

**Returns**: Complex tensor of shape (end, dim//2) containing rotary embedding frequencies.

#### `reshape_for_broadcast(freqs_cis, x)`

Reshapes the frequencies tensor for broadcasting with input tensors.

| Parameter | Type | Description |
|-----------|------|-------------|
| `freqs_cis` | torch.Tensor | Complex-valued frequencies from `compute_freqs_cis` |
| `x` | torch.Tensor | Input tensor to broadcast with |

**Returns**: Reshaped frequency tensor ready for broadcasting.

#### `apply_rope(xq, xk, freqs_cis)`

Applies rotary embeddings to query and key tensors.

| Parameter | Type | Description |
|-----------|------|-------------|
| `xq` | torch.Tensor | Query tensor of shape (batch, seq_len, n_heads, head_dim) |
| `xk` | torch.Tensor | Key tensor of shape (batch, seq_len, n_kv_heads, head_dim) |
| `freqs_cis` | torch.Tensor | Complex-valued rotary embeddings |

**Returns**: Tuple of (xq, xk) with positional information applied.

### Example

```python
import torch
from tlama_core.models import compute_freqs_cis, apply_rope

# Generate rotary embeddings
max_seq_len = 2048
head_dim = 64
freqs_cis = compute_freqs_cis(head_dim, max_seq_len)

# Apply to query and key tensors
batch_size = 4
seq_len = 512
n_heads = 16
n_kv_heads = 4

xq = torch.randn(batch_size, seq_len, n_heads, head_dim)
xk = torch.randn(batch_size, seq_len, n_kv_heads, head_dim)

xq_rope, xk_rope = apply_rope(xq, xk, freqs_cis[:seq_len])
```

---

## Key-Value Repetition

For models that use grouped-query attention (where n_kv_heads < n_heads), the key-value heads need to be repeated.

### Functions

#### `repeat_kv(kv, n_rep)`

Repeats key or value heads to match the number of query heads.

| Parameter | Type | Description |
|-----------|------|-------------|
| `kv` | torch.Tensor | Key or value tensor of shape (batch, seq_len, n_kv_heads, head_dim) |
| `n_rep` | int | Number of repetitions needed (n_heads / n_kv_heads) |

**Returns**: Tensor with repeated heads of shape (batch, seq_len, n_kv_heads*n_rep, head_dim).

### Example

```python
import torch
from tlama_core.models import repeat_kv

# Create key tensor
batch_size = 4
seq_len = 512
n_kv_heads = 4
head_dim = 64
key = torch.randn(batch_size, seq_len, n_kv_heads, head_dim)

# Repeat to match 16 query heads (n_rep = 4)
key_repeated = repeat_kv(key, n_rep=4)
# Result shape: (4, 512, 16, 64)
```

---

## Attention Mechanism

The `Attention` module implements multi-head attention with optional key-value caching and rotary positional embeddings.

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `config` | TlamaConfig | Configuration object with model hyperparameters |

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `n_kv_heads` | int | Number of key-value heads |
| `n_local_heads` | int | Number of local query heads (for model parallelism) |
| `n_local_kv_heads` | int | Number of local key-value heads |
| `n_rep` | int | Repetition factor for key-value heads |
| `head_dim` | int | Dimension of each attention head |
| `Wq`, `Wk`, `Wv` | Linear/ColumnParallelLinear | Linear projections for query, key, and value |
| `Wo` | Linear/RowParallelLinear | Output projection |
| `cache_k`, `cache_v` | torch.Tensor | Caches for keys and values (if enabled) |

### Methods

#### `forward(x, start_pos, freq_cis, mask, kv_cache=True)`

Performs the attention operation.

| Parameter | Type | Description |
|-----------|------|-------------|
| `x` | torch.Tensor | Input tensor of shape (batch_size, seq_len, d_model) |
| `start_pos` | int | Starting position in the sequence (for caching) |
| `freq_cis` | torch.Tensor | Rotary positional embeddings |
| `mask` | Optional[torch.Tensor] | Attention mask tensor |
| `kv_cache` | bool, optional | Whether to use key-value caching (default: True) |

**Returns**: Output tensor after applying attention.

### Example

```python
from tlama_core.models import Attention
from tlama_core.models.config import TlamaConfig

# Configuration
config = TlamaConfig(
    d_model=1024,
    n_heads=16,
    n_kv_heads=4,  # Grouped-query attention
    max_batch_size=32,
    max_seq_len=2048,
    kv_cache=True
)

# Create attention module
attention = Attention(config)

# Forward pass
batch_size = 4
seq_len = 128
x = torch.randn(batch_size, seq_len, config.d_model)
start_pos = 0
freq_cis = compute_freqs_cis(config.d_model // config.n_heads, seq_len)
mask = None  # For autoregressive generation, use causal mask

output = attention(x, start_pos, freq_cis, mask)
```

---

## FeedForward Network

The `FeedForward` module implements a SwiGLU-based feed-forward network with configurable dimensions.

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `d_model` | int | Model dimension (input and output) |
| `hidden_dim` | int | Hidden dimension (before adjustment) |
| `multiple_of` | int | Ensures hidden_dim is a multiple of this value |
| `ffn_dim_multiplier` | Optional[float] | Multiplier for feed-forward dimension |
| `use_parallel` | bool, optional | Whether to use model parallelism (default: True) |

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `w1` | Linear/ColumnParallelLinear | First projection |
| `w2` | Linear/RowParallelLinear | Output projection |
| `w3` | Linear/ColumnParallelLinear | Gate projection for SwiGLU |

### Methods

#### `forward(x)`

Applies the feed-forward network.

| Parameter | Type | Description |
|-----------|------|-------------|
| `x` | torch.Tensor | Input tensor of shape (batch_size, seq_len, d_model) |

**Returns**: Output tensor after applying the feed-forward network.

### Example

```python
import torch
from tlama_core.models import FeedForward

# Create feed-forward network
ff_network = FeedForward(
    d_model=1024,
    hidden_dim=4096,
    multiple_of=256,
    ffn_dim_multiplier=1.0,
    use_parallel=False
)

# Forward pass
x = torch.randn(4, 128, 1024)  # (batch_size, seq_len, d_model)
output = ff_network(x)
```

---

## Transformer Block

The `TransformerBlock` combines attention and feed-forward modules with residual connections and layer normalization.

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `layer_id` | int | Layer identifier |
| `config` | TlamaConfig | Configuration object with model hyperparameters |

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `attention` | Attention | Attention module |
| `ffn` | FeedForward | Feed-forward network |
| `attention_norm` | RMSNorm | Layer normalization before attention |
| `ffn_norm` | RMSNorm | Layer normalization before feed-forward |

### Methods

#### `forward(x, start_pos, freq_cis, mask)`

Applies the transformer block.

| Parameter | Type | Description |
|-----------|------|-------------|
| `x` | torch.Tensor | Input tensor of shape (batch_size, seq_len, d_model) |
| `start_pos` | int | Starting position in the sequence |
| `freq_cis` | torch.Tensor | Rotary positional embeddings |
| `mask` | Optional[torch.Tensor] | Attention mask tensor |

**Returns**: Output tensor after applying the transformer block.

### Example

```python
import torch
from tlama_core.models import TransformerBlock
from tlama_core.models.config import TlamaConfig

# Configuration
config = TlamaConfig(
    d_model=1024,
    n_heads=16,
    n_kv_heads=4,
    ffn_dim_multiplier=1.0,
    multiple_of=256,
    norm_eps=1e-5
)

# Create transformer block
block = TransformerBlock(layer_id=0, config=config)

# Forward pass
batch_size = 4
seq_len = 128
x = torch.randn(batch_size, seq_len, config.d_model)
start_pos = 0
freq_cis = compute_freqs_cis(config.d_model // config.n_heads, seq_len)
mask = None  # For autoregressive generation, use causal mask

output = block(x, start_pos, freq_cis, mask)
```

---

## Transformer Model

The `Transformer` class implements the complete transformer architecture.

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `config` | TlamaConfig | Configuration object with model hyperparameters |

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `token_emb` | Embedding/VocabParallelEmbedding | Token embedding layer |
| `layers` | ModuleList | List of transformer blocks |
| `norm` | RMSNorm | Final layer normalization |
| `output` | Linear/ColumnParallelLinear | Output projection to vocabulary |
| `freq_cis` | torch.Tensor | Precomputed rotary positional embeddings |

### Methods

#### `forward(tokens, start_pos, targets=None)`

Performs a forward pass through the transformer.

| Parameter | Type | Description |
|-----------|------|-------------|
| `tokens` | torch.Tensor | Input token IDs of shape (batch_size, seq_len) |
| `start_pos` | int | Starting position in the sequence |
| `targets` | Optional[torch.Tensor] | Target token IDs for computing loss |

**Returns**: Tuple of (outputs, loss) where loss is None if targets is None.

#### `_init_weights(module)`

Initializes the weights of the model's modules.

| Parameter | Type | Description |
|-----------|------|-------------|
| `module` | nn.Module | Module to initialize |

#### `configure_optimizers(weight_decay, learning_rate, device_type, master_process=True)`

Configures optimizers with appropriate parameter groups.

| Parameter | Type | Description |
|-----------|------|-------------|
| `weight_decay` | float | Weight decay coefficient |
| `learning_rate` | float | Learning rate |
| `device_type` | str | Device type ('cuda', 'cpu', etc.) |
| `master_process` | bool, optional | Whether this is the master process (default: True) |

**Returns**: Configured optimizer.

### Example

```python
import torch
from tlama_core.models import Transformer
from tlama_core.models.config import TlamaConfig

# Configuration
config = TlamaConfig(
    vocab_size=32000,
    d_model=1024,
    n_layers=12,
    n_heads=16,
    n_kv_heads=4,
    max_seq_len=2048,
    max_batch_size=32,
    norm_eps=1e-5,
    rope_theta=10000.0,
    weight_init=[0.02, 0.0],  # std, mean
    use_parallel=False,
    kv_cache=True
)

# Create model
model = Transformer(config)

# Forward pass
batch_size = 4
seq_len = 128
tokens = torch.randint(0, config.vocab_size, (batch_size, seq_len))
start_pos = 0
targets = tokens.clone()  # For training

logits, loss = model(tokens, start_pos, targets)

# Configure optimizer
optimizer = model.configure_optimizers(
    weight_decay=0.1,
    learning_rate=1e-4,
    device_type='cuda'
)

# Training step
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

## Integration with Training Loop

Here's an example of how to integrate the model into a training loop:

```python
from tlama_core.models import Transformer
from tlama_core.models.config import TlamaConfig
from tlama_core.training import get_lr_scheduler
import torch

# Configuration
config = TlamaConfig(
    vocab_size=32000,
    d_model=1024,
    n_layers=12,
    n_heads=16,
    n_kv_heads=4,
    max_seq_len=2048,
    max_batch_size=32
)

# Create model
model = Transformer(config)

# Optimizer
optimizer = model.configure_optimizers(
    weight_decay=0.1,
    learning_rate=0.0,  # Will be set by scheduler
    device_type='cuda'
)

# Learning rate scheduler
scheduler = get_lr_scheduler(
    max_lr=1e-4,
    min_lr=1e-6,
    warmup_steps=1000,
    max_steps=50000
)

# Training loop
for step, batch in enumerate(dataloader):
    # Update learning rate
    lr = scheduler(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    # Get inputs and targets
    input_ids = batch['input_ids'].to(device)
    targets = batch['targets'].to(device)
    
    # Forward pass
    logits, loss = model(input_ids, start_pos=0, targets=targets)
    
    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Logging
    if step % 100 == 0:
        print(f"Step {step}, Loss: {loss.item()}, LR: {lr}")
```