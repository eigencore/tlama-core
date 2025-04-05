# Trainer Module

The `Trainer` class in the `tlama-core` library provides a comprehensive training loop for training transformer models. It handles all aspects of the training process including gradient accumulation, mixed-precision training, checkpointing, logging, and validation.

## Trainer Class

The `Trainer` class manages the entire training process for Tlama models, with support for various optimization strategies and training workflows.

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | torch.nn.Module | The model to train |
| `train_data_loader` | Any | DataLoader for training data |
| `optimizer` | Optional[torch.optim.Optimizer] | Optimizer to use (if None, will be created from model) |
| `scheduler` | Optional[Callable] | Learning rate scheduler (function that takes step and returns lr) |
| `val_data_loader` | Optional[Any] | DataLoader for validation data |
| `total_batch_size` | int | Total batch size in tokens (default: 524288, ~0.5M tokens) |
| `gradient_accumulation` | bool | Whether to use gradient accumulation (default: True) |
| `weight_decay` | float | Weight decay for optimizer (default: 0.1) |
| `learning_rate` | float | Learning rate if no scheduler is provided (default: 6e-4) |
| `epochs` | int | Number of epochs to train (default: 1) |
| `steps` | Optional[int] | Number of steps per epoch (if None, determined by data_loader) |
| `validation_steps` | int | Run validation every N steps (default: 250) |
| `checkpoint_steps` | int | Save checkpoint every N steps (default: 5000) |
| `log_steps` | int | Log metrics every N steps (default: 10) |
| `gradient_clip_val` | float | Max norm for gradient clipping (default: 1.0) |
| `use_mixed_precision` | bool | Whether to use mixed precision training (default: True) |
| `checkpoints_dir` | str | Directory to save checkpoints (default: "checkpoints") |
| `logs_dir` | str | Directory to save logs (default: "logs") |
| `seed` | int | Random seed for reproducibility (default: 1337) |
| `verbose` | bool | Whether to print training information (default: True) |
| `callbacks` | List[Any] | List of callbacks to use during training (default: None) |
| `master_process` | bool | Whether this is the master process for distributed training (default: True) |

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `model` | torch.nn.Module | The model being trained |
| `optimizer` | torch.optim.Optimizer | Optimizer instance |
| `scheduler` | Optional[Callable] | Learning rate scheduler |
| `device` | torch.device | Device used for training |
| `device_type` | str | Type of device ('cuda', 'mps', or 'cpu') |
| `grad_accum_steps` | int | Number of gradient accumulation steps |
| `current_epoch` | int | Current training epoch |
| `current_step` | int | Current global training step |
| `best_val_loss` | float | Best validation loss observed |
| `log_file` | str | Path to the log file |

### Methods

#### `__init__(...)`

Initializes the Trainer with specified parameters.

#### `_set_seed(seed)`

Sets random seeds for reproducibility.

| Parameter | Type | Description |
|-----------|------|-------------|
| `seed` | int | Random seed to set |

#### `_log_metrics(metrics)`

Logs metrics to the log file.

| Parameter | Type | Description |
|-----------|------|-------------|
| `metrics` | Dict[str, Any] | Dictionary of metrics to log |

#### `_save_checkpoint(val_loss=None, is_best=False)`

Saves a model checkpoint.

| Parameter | Type | Description |
|-----------|------|-------------|
| `val_loss` | Optional[float] | Validation loss to include in checkpoint |
| `is_best` | bool | Whether this is the best model so far |

#### `validate()`

Runs validation on the model.

**Returns**: Optional[float] - Validation loss or None if no validation data

#### `train_step()`

Runs a single training step (with gradient accumulation).

**Returns**: Dict[str, float] - Metrics from the training step

#### `train()`

Runs the full training loop.

**Returns**: Dict[str, Any] - Final training metrics

### Example Usage

Here's a complete example of how to use the Trainer class:

```python
import torch
from tlama_core.models import Transformer
from tlama_core.models.config import TlamaConfig
from tlama_core.training import Trainer, get_lr_scheduler
from tlama_core.data import create_dataloader

# Create model
config = TlamaConfig(
    vocab_size=32000,
    d_model=1024,
    n_layers=12,
    n_heads=16,
    n_kv_heads=4,
    max_seq_len=2048,
    max_batch_size=32
)
model = Transformer(config)

# Create data loaders
train_loader = create_dataloader(
    data_path="training_data.txt",
    tokenizer_path="tokenizer.model",
    batch_size=32,
    seq_length=2048,
    shuffle=True
)

val_loader = create_dataloader(
    data_path="validation_data.txt",
    tokenizer_path="tokenizer.model",
    batch_size=32,
    seq_length=2048,
    shuffle=False
)

# Create learning rate scheduler
scheduler = get_lr_scheduler(
    max_lr=1e-4,
    min_lr=1e-6,
    warmup_steps=1000,
    max_steps=50000,
    decay_type="cosine"
)

# Create trainer
trainer = Trainer(
    model=model,
    train_data_loader=train_loader,
    val_data_loader=val_loader,
    scheduler=scheduler,
    total_batch_size=1048576,  # ~1M tokens per batch
    gradient_accumulation=True,
    epochs=3,
    validation_steps=500,
    checkpoint_steps=2000,
    use_mixed_precision=True,
    checkpoints_dir="./checkpoints",
    logs_dir="./logs",
    seed=42,
    verbose=True
)

# Run training
results = trainer.train()
print(f"Training completed. Best validation loss: {results['best_val_loss']}")
```

## Callback System

The Trainer class supports a flexible callback system that allows you to hook into various stages of the training process.

### Available Callback Hooks
> **⚠️ Warning:**  
> The callback system is not yet implemented.

| Hook | Description |
|------|-------------|
| `on_init_end` | Called at the end of trainer initialization |
| `on_train_start` | Called at the beginning of training |
| `on_train_end` | Called at the end of training |
| `on_epoch_start` | Called at the beginning of each epoch |
| `on_epoch_end` | Called at the end of each epoch |
| `on_step_start` | Called at the beginning of each step |
| `on_step_end` | Called at the end of each step, receives metrics |
| `on_backward_start` | Called before the backward pass, receives loss |
| `on_backward_end` | Called after the backward pass |
| `on_optimizer_step_start` | Called before optimizer step |
| `on_optimizer_step_end` | Called after optimizer step |
| `on_validate_start` | Called at the beginning of validation |
| `on_validate_end` | Called at the end of validation, receives metrics |
| `on_checkpoint_end` | Called after saving a checkpoint, receives path |

### Creating a Custom Callback

```python
class LoggingCallback:
    def on_train_start(self, trainer):
        print("Training started!")
        
    def on_step_end(self, trainer, metrics):
        if trainer.current_step % 100 == 0:
            print(f"Step {trainer.current_step}: Loss = {metrics['train_loss']:.4f}")
    
    def on_epoch_end(self, trainer):
        print(f"Epoch {trainer.current_epoch + 1} completed")
    
    def on_validate_end(self, trainer, metrics):
        print(f"Validation loss: {metrics['val_loss']:.4f}")

# Use the callback with the trainer
callbacks = [LoggingCallback()]
trainer = Trainer(model=model, train_data_loader=train_loader, callbacks=callbacks)
```

## Gradient Accumulation

The Trainer automatically handles gradient accumulation to enable training with large effective batch sizes.

The number of gradient accumulation steps is calculated as:
```
grad_accum_steps = total_batch_size / (batch_size * seq_len)
```

Where:
- `total_batch_size` is the desired total number of tokens per batch
- `batch_size` is the mini-batch size of the dataloader
- `seq_len` is the sequence length of each sample

For example, if you want to train with a total batch size of 1M tokens, but your GPU can only handle a mini-batch of 8 samples with 2048 sequence length (16,384 tokens), the Trainer will automatically use 61 gradient accumulation steps.

## Mixed Precision Training

When `use_mixed_precision=True` and training on CUDA devices, the Trainer automatically uses PyTorch's automatic mixed precision with bfloat16. This can significantly speed up training while maintaining numerical stability.

## Checkpointing and Logging

The Trainer saves checkpoints at regular intervals (defined by `checkpoint_steps`) and logs training metrics to a CSV file in the `logs_dir` directory. The best model (lowest validation loss) is also saved in a separate checkpoint file.

Each checkpoint contains:
- Model state dictionary
- Optimizer state
- Current epoch and step
- Validation loss (if available)
- Model configuration (if available)

The log file includes the following metrics for each logged step:
- Epoch and step number
- Timestamp
- Training loss
- Learning rate
- Gradient norm
- Validation loss (when available)
- Tokens per second (throughput)