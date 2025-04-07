# 🚀 Training Tlama Models: A Complete Tutorial
<div align="center">
  <img src="https://i.postimg.cc/R00W9YMj/Tlama-1.png" alt="EigenCore" width="150" style="margin-right: 50px;">
</div>
The mission of EigenCore and Tlama is to provide a high-performance, open-source framework for training and deploying large language models (LLMs).

This tutorial will guide you through the process of training a Tlama model using Tlama-Core, the library developed by the EigenCore team.
📑 Table of Contents
<table data-card-size="large" data-view="cards">
<thead>
<tr>
<th></th>
<th></th>
</tr>
</thead>
<tbody>
<tr>
<td><strong>🔧 Installation</strong></td>
<td>Set up the Tlama-Core library and its dependencies</td>
</tr>
<tr>
<td><strong>📊 Data Preparation</strong></td>
<td>Prepare your dataset for training</td>
</tr>
<tr>
<td><strong>⚙️ Model Configuration</strong></td>
<td>Configure the model architecture and parameters</td>
</tr>
<tr>
<td><strong>🔄 Training Process</strong></td>
<td>Train and monitor your model</td>
</tr>
<tr>
<td><strong>📈 Evaluation</strong></td>
<td>Evaluate your model's performance</td>
</tr>
<tr>
<td><strong>💾 Model Management</strong></td>
<td>Save and load your trained models</td>
</tr>
<tr>
<td><strong>🔍 Troubleshooting</strong></td>
<td>Solve common training issues</td>
</tr>
</tbody>
</table>

# 🔧 Installation

To install Tlama-Core, you can use pip. Make sure you have Python 3.8 or higher installed. First, create a virtual environment and activate it:

```bash
python -m venv tlamacore-env
source tlamacore-env/bin/activate  # On Windows use `tlamacore-env\Scripts\activate`
```

Then, install Tlama-Core using pip:

```bash
pip install tlama-core
```

If you encounter dependency issues, you can install them manually with pip.

## 📊 Data Preparation

For this tutorial, we'll use the open-source `fineweb-edu` dataset from [Hugging Face](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu). Let's create a directory for our data and a script to download and process the dataset:

```bash
mkdir data
cd data
touch download_dataset.py
```

The download_dataset.py script will:
1. Download the dataset from Hugging Face
2. Tokenize the text using the GPT-2 tokenizer
3. Save the tokenized data in shards for efficient training

Here's the complete script:

```python
import os
import multiprocessing as mp
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

# Configuration
CONFIG = {
    "local_dir": "edu_fineweb10B",
    "remote_name": "sample-10BT",
    "shard_size": int(1e8),  # 100M tokens per shard
    "num_processes": max(1, os.cpu_count()//2)
}

def setup_directories(config):
    """Creates the local directory to store shards."""
    data_cache_dir = os.path.join(os.path.dirname(__file__), config["local_dir"])
    os.makedirs(data_cache_dir, exist_ok=True)
    return data_cache_dir

def load_fineweb_dataset(config):
    """Loads dataset from HuggingFace."""
    return load_dataset("HuggingFaceFW/fineweb-edu", 
                       name=config["remote_name"], 
                       split="train")

def initialize_tokenizer():
    """Initializes the tiktoken tokenizer."""
    enc = tiktoken.get_encoding("gpt2")
    eot = enc._special_tokens['<|endoftext|>']
    return enc, eot

def tokenize(doc, enc=None, eot=None):
    """
    Tokenizes a document and returns a numpy array of uint16 tokens.
    """
    if enc is None or eot is None:
        enc, eot = initialize_tokenizer()
        
    # Tokenize with EOT token at the beginning
    tokens = [eot]
    tokens.extend(enc.encode_ordinary(doc["text"]))
    
    # Convert to numpy array of uint16
    tokens_np = np.array(tokens)
    assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), "token dictionary too large for uint16"
    return tokens_np.astype(np.uint16)

def write_datafile(filename, tokens_np):
    """Saves tokens to a numpy file."""
    np.save(filename, tokens_np)

def process_shard(tokens, shard_buffer, token_count, shard_size, data_dir, shard_index, progress_bar=None):
    """
    Processes a tokenized document, adding it to the current shard or creating a new one.
    """
    # Is there enough space in the current shard for the new tokens?
    if token_count + len(tokens) < shard_size:
        # Simply append tokens to current shard
        shard_buffer[token_count:token_count+len(tokens)] = tokens
        token_count += len(tokens)
        
        # Update progress bar
        if progress_bar is None:
            progress_bar = tqdm(total=shard_size, unit="tokens", desc=f"Shard {shard_index}")
        progress_bar.update(len(tokens))
        
        return token_count, shard_index, progress_bar
    else:
        # Write the current shard and start a new one
        split = "val" if shard_index == 0 else "train"
        filename = os.path.join(data_dir, f"edufineweb_{split}_{shard_index:06d}")
        
        # Split document: part in this shard, remainder in next
        remainder = shard_size - token_count
        progress_bar.update(remainder)
        shard_buffer[token_count:token_count+remainder] = tokens[:remainder]
        write_datafile(filename, shard_buffer)
        
        # Start a new shard
        shard_index += 1
        progress_bar = None
        
        # Initialize next shard with remaining tokens
        shard_buffer[0:len(tokens)-remainder] = tokens[remainder:]
        token_count = len(tokens)-remainder
        
        return token_count, shard_index, progress_bar

def main():
    """Main function to process the dataset."""
    # Initial setup
    data_dir = setup_directories(CONFIG)
    dataset = load_fineweb_dataset(CONFIG)
    enc, eot = initialize_tokenizer()
    
    # Pre-initialize tokenizer for the pool
    tokenize_fn = lambda doc: tokenize(doc, enc, eot)
    
    # Create process pool for parallel tokenization
    with mp.Pool(CONFIG["num_processes"]) as pool:
        # Initialize variables for shard processing
        shard_index = 0
        shard_buffer = np.empty((CONFIG["shard_size"],), dtype=np.uint16)
        token_count = 0
        progress_bar = None
        
        # Process each tokenized document
        for tokens in pool.imap(tokenize_fn, dataset, chunksize=16):
            token_count, shard_index, progress_bar = process_shard(
                tokens, shard_buffer, token_count, CONFIG["shard_size"], 
                data_dir, shard_index, progress_bar
            )

        # Write any remaining tokens as the last shard
        if token_count > 0:
            split = "val" if shard_index == 0 else "train"
            filename = os.path.join(data_dir, f"edufineweb_{split}_{shard_index:06d}")
            write_datafile(filename, shard_buffer[:token_count])
            if progress_bar:
                progress_bar.close()
    
    print(f"Processing completed. Shards saved in {data_dir}")

if __name__ == "__main__":
    main()
```

Run the script to download and process the dataset:

```bash
python download_dataset.py
```

This will create a directory called `edu_fineweb10B` with tokenized shards of the dataset.

## ⚙️ Model Configuration
Now let's create a training script that will set up our model configuration, data loaders, and training process. Create a new file called `train.py` in your project directory:

```bash
touch train.py
```

Let's build our training script step by step:

First, we'll import the necessary modules and set up the model configuration:

```python
from tlamacore.models.config import TlamaConfig
from tlamacore.models.tlama_base import Transformer
from tlamacore.data.dataloader import DataLoaderLite
from tlamacore.training.scheduler import get_lr_scheduler
from tlamacore.training.trainer import Trainer

# Configuration for Tlama model
config = TlamaConfig(
    d_model=512,         # Hidden size
    n_layers=8,          # Number of transformer layers
    n_heads=8,           # Number of attention heads
    n_kv_heads=None,     # Number of key-value heads (None means equal to n_heads)
    vocab_size=50304,    # Vocabulary size
    multiple_of=256,     # Ensures hidden layer size in SwiGLU is a multiple of this value
    ffn_dim_multiplier=None,  # Multiplier for FFN dimension
    norm_eps=1e-5,       # Normalization epsilon
    rope_theta=10000.0,  # RoPE theta parameter
    max_batch_size=32,   # Maximum batch size
    max_seq_len=1024,    # Maximum sequence length
    use_parallel=False,  # Whether to use parallel attention
    kv_cache=False       # Whether to use KV cache
)

# Initialize the Transformer model
model = Transformer(config)
```

## DataLoader Setup

Next, we'll set up the data loaders for training and validation:

```python
# Training data loader
train_data_loader = DataLoaderLite(
    data_dir='edu_fineweb10B',  # Directory containing the sharded dataset
    batch_size=2,               # Batch size
    seq_len=1024,               # Sequence length
    process_rank=0,             # Process rank (for distributed training)
    num_process=1,              # Number of processes (for distributed training)
    split='train',              # Dataset split
    shuffle=True,               # Whether to shuffle the data
    verbose=False,              # Whether to show verbose output
)

# Validation data loader
val_data_loader = DataLoaderLite(
    data_dir='edu_fineweb10B',  # Directory containing the sharded dataset
    batch_size=2,               # Batch size
    seq_len=1024,               # Sequence length
    process_rank=0,             # Process rank (for distributed training)
    num_process=1,              # Number of processes (for distributed training)
    split='val',                # Dataset split
    shuffle=False,              # Don't shuffle validation data
    verbose=False,              # Whether to show verbose output
)
```

## Training Configuration

Now, let's set up the learning rate scheduler and trainer:

```python
# Learning rate scheduler
scheduler = get_lr_scheduler(
    max_lr=6e-4,           # Maximum learning rate
    min_lr=6e-4 * 0.1,     # Minimum learning rate
    max_steps=19073,       # Maximum number of training steps
    warmup_steps=715,      # Number of warmup steps
    decay_type='cosine',   # Type of learning rate decay
    verbose=False,         # Whether to show verbose output
)

# Trainer with advanced configuration
trainer = Trainer(
    model=model,                  # The model to train
    train_data_loader=train_data_loader,  # Training data loader
    optimizer=None,               # Optimizer (None will create a default AdamW)
    scheduler=scheduler,          # Learning rate scheduler
    val_data_loader=val_data_loader,      # Validation data loader
    total_batch_size=524288,      # Total batch size (~0.5M tokens)
    gradient_accumulation=True,   # Whether to use gradient accumulation
    weight_decay=0.1,             # Weight decay for AdamW optimizer
    learning_rate=6e-4,           # Learning rate (if scheduler is None)
    epochs=1,                     # Number of epochs
    steps=19073,                  # Number of steps
    validation_steps=250,         # Run validation every N steps
    checkpoint_steps=5000,        # Save checkpoint every N steps
    log_steps=10,                 # Log metrics every N steps
    gradient_clip_val=1.0,        # Clip gradients to this norm
    use_mixed_precision=True,     # Whether to use mixed precision training
    checkpoints_dir="checkpoints", # Directory to save checkpoints
    logs_dir="logs",              # Directory to save logs
    seed=1337,                    # Random seed
    verbose=True                  # Whether to print training information
)
```

## Training the Model

Finally, let's start the training process:

```python
# Start training
trainer.train()
```

Let's put everything together in our `train.py` file:

```python
from tlamacore.data.dataloader import DataLoaderLite
from tlamacore.training.scheduler import get_lr_scheduler
from tlamacore.training.trainer import Trainer
from tlamacore.models.tlama_base import Transformer
from tlamacore.models.config import TlamaConfig

# Configuration for Tlama model
config = TlamaConfig(
    d_model=512,         # Hidden size
    n_layers=8,          # Number of transformer layers
    n_heads=8,           # Number of attention heads
    n_kv_heads=None,     # Number of key-value heads
    vocab_size=50304,    # Vocabulary size
    multiple_of=256,     # Ensures hidden layer size in SwiGLU is a multiple of this value
    ffn_dim_multiplier=None,  # Multiplier for FFN dimension
    norm_eps=1e-5,       # Normalization epsilon
    rope_theta=10000.0,  # RoPE theta parameter
    max_batch_size=32,   # Maximum batch size
    max_seq_len=1024,    # Maximum sequence length
    use_parallel=False,  # Whether to use parallel attention
    kv_cache=False       # Whether to use KV cache
)

# Initialize the Transformer model
model = Transformer(config)

# Training data loader
train_data_loader = DataLoaderLite(
    data_dir='edu_fineweb10B',  # Directory containing the sharded dataset
    batch_size=2,               # Batch size
    seq_len=1024,               # Sequence length
    process_rank=0,             # Process rank (for distributed training)
    num_process=1,              # Number of processes (for distributed training)
    split='train',              # Dataset split
    shuffle=True,               # Whether to shuffle the data
    verbose=False,              # Whether to show verbose output
)

# Validation data loader
val_data_loader = DataLoaderLite(
    data_dir='edu_fineweb10B',  # Directory containing the sharded dataset
    batch_size=2,               # Batch size
    seq_len=1024,               # Sequence length
    process_rank=0,             # Process rank (for distributed training)
    num_process=1,              # Number of processes (for distributed training)
    split='val',                # Dataset split
    shuffle=False,              # Don't shuffle validation data
    verbose=False,              # Whether to show verbose output
)

# Learning rate scheduler
scheduler = get_lr_scheduler(
    max_lr=6e-4,           # Maximum learning rate
    min_lr=6e-4 * 0.1,     # Minimum learning rate
    max_steps=19073,       # Maximum number of training steps
    warmup_steps=715,      # Number of warmup steps
    decay_type='cosine',   # Type of learning rate decay
    verbose=False,         # Whether to show verbose output
)

# Trainer
trainer = Trainer(
    model=model,                  # The model to train
    scheduler=scheduler,          # Learning rate scheduler
    train_data_loader=train_data_loader,  # Training data loader
    val_data_loader=val_data_loader,      # Validation data loader
    epochs=1,                     # Number of epochs
    steps=19073,                  # Number of steps
    log_steps=1                   # Log frequency
)

# Start training
trainer.train()
```

Now you can run the training script:

```bash
python train.py
```

## Monitoring Training

Tlama-Core's Trainer class provides built-in functionality for monitoring the training process. As training progresses, it will output metrics like loss, learning rate, gradient norm, and validation performance at the specified log intervals.

By default, the Trainer will:
- Log metrics to a file in the `logs_dir` directory (default: "logs/training_log.txt")
- Print metrics to the console every `log_steps` steps (default: 10)
- Save model checkpoints to the `checkpoints_dir` directory (default: "checkpoints")
- Run validation every `validation_steps` steps (default: 250)

Here's a sample of the output you might see during training:

```
Tlama-Core: INFO: Using device: cuda
Tlama-Core: INFO: Gradient accumulation enabled with 128 steps.
Tlama-Core: INFO: Creating optimizer.
Tlama-Core: INFO: Setting seed to 1337
Tlama-Core: INFO: Starting epoch 1/1
Step   10 | loss: 11.329548 | lr: 6.0000e-05 | norm: 1.0000 | dt: 450.32ms | tok/sec: 1165.18
Step   20 | loss: 11.028763 | lr: 1.2000e-04 | norm: 1.0000 | dt: 448.73ms | tok/sec: 1169.08
Tlama-Core: INFO: Validation loss: 10.9843
```

For more sophisticated monitoring, you can add custom callbacks or use TensorBoard integration. The Trainer class supports a callback interface for monitoring and modifying the training process. Before starting training, add:

```python
from tlamacore.training.callback import TensorboardCallback, CallbackBase

# Create TensorBoard callback for visualization
tb_callback = TensorboardCallback(log_dir='./logs')

# You can also create custom callbacks by extending CallbackBase
class CustomCallback(CallbackBase):
    def on_step_end(self, trainer, metrics):
        # Access metrics after each step
        if trainer.current_step % 100 == 0:
            print(f"Custom metric at step {trainer.current_step}: {metrics['train_loss']}")

# Add callbacks to the trainer
trainer = Trainer(
    model=model,
    train_data_loader=train_data_loader,
    val_data_loader=val_data_loader,
    callbacks=[tb_callback, CustomCallback()],  # Add callbacks here
    # ... other parameters ...
)
```

Then you can start TensorBoard to visualize your training metrics:

```bash
tensorboard --logdir=./logs
```

The Trainer also automatically creates a CSV log file that records training progress, which you can analyze after training using pandas or other data analysis tools:

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load training log
training_log = pd.read_csv("logs/training_log.txt")

# Plot training loss
plt.figure(figsize=(10, 6))
plt.plot(training_log['step'], training_log['train_loss'])
plt.xlabel('Step')
plt.ylabel('Loss')
plt.title('Training Loss Over Time')
plt.savefig('training_loss.png')
```

## Saving and Loading the Model

The Trainer automatically saves checkpoints during training:
- Regular checkpoints every `checkpoint_steps` steps (default: 5000)
- The best checkpoint based on validation loss
- A final checkpoint at the end of training

These checkpoints are saved to the `checkpoints_dir` directory (default: "checkpoints"). Each checkpoint contains:
- Model state dict
- Optimizer state dict
- Training epoch and step
- Model configuration (if available)
- Validation loss (if available)

To manually save a checkpoint at any point during or after training:

```python
# Save a checkpoint
trainer._save_checkpoint()

# Or save with validation loss
val_loss = trainer.validate()
trainer._save_checkpoint(val_loss=val_loss, is_best=True)
```

To load a saved checkpoint for inference or continued training:

```python
import torch
from tlamacore.models.tlama_base import Transformer
from tlamacore.models.config import TlamaConfig

# Load the checkpoint
checkpoint_path = 'checkpoints/checkpoint_best.pt'
checkpoint = torch.load(checkpoint_path)

# Create a model with the saved configuration
if 'config' in checkpoint:
    # If config was saved in the checkpoint
    config = checkpoint['config']
    model = Transformer(config)
else:
    # Otherwise, recreate the configuration
    config = TlamaConfig(
        d_model=512,
        n_layers=8,
        n_heads=8,
        # ... other parameters ...
    )
    model = Transformer(config)

# Load the model weights
model.load_state_dict(checkpoint['model'])

# Optionally, restore optimizer state for continued training
optimizer = torch.optim.AdamW(model.parameters())
optimizer.load_state_dict(checkpoint['optimizer'])

# Get training progress information
current_epoch = checkpoint.get('epoch', 0)
current_step = checkpoint.get('step', 0)
best_val_loss = checkpoint.get('val_loss', float('inf'))

print(f"Loaded checkpoint from epoch {current_epoch}, step {current_step}")
if 'val_loss' in checkpoint:
    print(f"Validation loss: {best_val_loss:.4f}")
```


## Evaluating the Model

After training, you might want to evaluate your model on various benchmarks or test sets. Here's a simple example of how to evaluate your model on a custom dataset:

```python
import torch
import numpy as np
from tlamacore.data.dataloader import DataLoaderLite

# Create a test data loader
test_data_loader = DataLoaderLite(
    data_dir='path/to/test/data',
    batch_size=1,
    seq_len=1024,
    process_rank=0,
    num_process=1,
    split='test',
    shuffle=False,
    verbose=False,
)

# Set model to evaluation mode
model.eval()

# Loop through test data
total_loss = 0
with torch.no_grad():
    for batch_idx, (x, y) in enumerate(test_data_loader):
        # Forward pass
        logits = model(x)
        
        # Calculate loss
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), 
            y.view(-1)
        )
        
        total_loss += loss.item()

# Calculate average loss
avg_loss = total_loss / len(test_data_loader)
perplexity = np.exp(avg_loss)

print(f"Test Loss: {avg_loss:.4f}")
print(f"Test Perplexity: {perplexity:.4f}")
```
<!-- 
## Fine-tuning the Model

You can also fine-tune your model on a specific dataset. Here's how:

```python
from tlamacore.training.scheduler import get_lr_scheduler
from tlamacore.training.trainer import Trainer
from tlamacore.data.dataloader import DataLoaderLite

# Load your pre-trained model
# ...

# Set up finetune data loader
finetune_data_loader = DataLoaderLite(
    data_dir='path/to/finetune/data',
    batch_size=2,
    seq_len=1024,
    process_rank=0,
    num_process=1,
    split='train',
    shuffle=True,
    verbose=False,
)

# Learning rate scheduler for fine-tuning (lower learning rate)
finetune_scheduler = get_lr_scheduler(
    max_lr=1e-5,           # Lower max learning rate for fine-tuning
    min_lr=1e-6,           # Lower min learning rate
    max_steps=1000,        # Fewer steps for fine-tuning
    warmup_steps=100,      # Fewer warmup steps
    decay_type='cosine',
    verbose=False,
)

# Create trainer for fine-tuning
finetune_trainer = Trainer(
    model=model,
    scheduler=finetune_scheduler,
    train_data_loader=finetune_data_loader,
    val_data_loader=None,  # Optionally, include a validation data loader
    epochs=1,
    steps=1000,
    log_steps=10
)

# Start fine-tuning
finetune_trainer.train()
``` -->

## Troubleshooting

Here are some common issues you might encounter during training and how to resolve them:

### Out of Memory Errors

If you encounter out of memory errors, try:
- Reducing batch size in the DataLoader
- Reducing sequence length
- Increasing gradient accumulation steps by setting a higher `total_batch_size` while keeping the DataLoader batch size small
- Enabling mixed precision training (on by default in Tlama-Core)

```python
# Configure for lower memory usage
trainer = Trainer(
    # ... other parameters ...
    total_batch_size=262144,      # Half the default (0.25M tokens)
    use_mixed_precision=True       # Enable mixed precision (default: True)
)
```

### Slow Training

If training is too slow, consider:
- Using a more powerful GPU
- Increasing batch size (if memory allows)
- Optimizing data loading (check if there are bottlenecks)
- Using a smaller model for initial experiments

You can profile your training loop to identify bottlenecks:

```python
# Add timing to identify slowdowns
start_time = time.time()
last_time = start_time

for step in range(100):
    # Run a training step
    metrics = trainer.train_step()
    
    current_time = time.time()
    print(f"Step {step} took {(current_time - last_time) * 1000:.2f}ms")
    last_time = current_time

print(f"Total time: {time.time() - start_time:.2f}s")
```

### NaN Losses

If you encounter NaN losses during training:
- Check for exploding gradients (reduce learning rate)
- Check for incorrect data normalization
- Try a different initialization method
- Add or reduce gradient clipping (default in Tlama-Core is 1.0)

```python
# Adjust gradient clipping for stability
trainer = Trainer(
    # ... other parameters ...
    gradient_clip_val=0.5  # Use a smaller clip value for stability
)
```

### Common Errors and Solutions

#### DataLoader issues
```
ValueError: DataLoader must have batch_size and seq_length/seq_len attributes
```
Make sure your DataLoader class exposes both `batch_size` and either `seq_length` or `seq_len` as attributes, not just as initialization parameters.

#### Validation errors
If validation is failing while training works:
```python
# Disable validation temporarily for debugging
trainer = Trainer(
    # ... other parameters ...
    val_data_loader=None,
    validation_steps=float('inf')  # Effectively disable validation
)
```

#### Model compatibility
If you see unexpected model behavior:
- Ensure your model returns both logits and loss in its forward pass
- Make sure loss reduction is set to 'mean'
- Check that your optimizer is compatible with the model parameters

#### CUDA out of memory during validation
If you're running out of memory during validation but not training:
```python
# Reduce validation batch size
val_data_loader = DataLoaderLite(
    # ... other parameters ...
    batch_size=1,  # Smaller batch for validation
)
```

## Conclusion

Awesome — you made it! 🎉  
Now you know how to train a Tlama model using the Tlama-Core library. We walked through the full process: getting your data ready, setting up the model, training, evaluating, and fine-tuning. From here, feel free to play around with different configs, datasets, and training tricks to see what works best for you.

Be sure to check out the Tlama-Core docs if you want to dive into more advanced stuff — there's plenty more to explore.

---

### Note:

Tlama-Core is still a work in progress, so if you run into any bugs or weird behavior, we’d really appreciate it if you let us know! We’re actively building and improving the library, and we'd love to have you join the community as it grows.

Thanks for being part of the journey 🚀

