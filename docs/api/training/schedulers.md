# Learning Rate Schedulers

Learning rate schedulers are essential components in training deep learning models as they control how the learning rate changes during training. The `tlama-core` library provides several built-in learning rate schedulers to improve convergence and training stability.

## Available Schedulers

- [`get_lr_scheduler`](#get_lr_scheduler): General-purpose scheduler with multiple decay types
- [`get_combined_lr_scheduler`](#get_combined_lr_scheduler): Combines multiple schedulers over different step ranges
- [`get_multi_step_lr_scheduler`](#get_multi_step_lr_scheduler): Implements a multi-step decay schedule

## Visualization Tools

- [`plot_lr_schedule`](#plot_lr_schedule): Visualizes any learning rate schedule

---

## get_lr_scheduler

The `get_lr_scheduler` function creates a learning rate scheduler with warmup and various decay options.

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `max_lr` | float | Maximum learning rate after warmup |
| `min_lr` | float | Minimum learning rate at the end of training |
| `warmup_steps` | int | Number of steps for linear warmup |
| `max_steps` | int | Total number of training steps |
| `decay_type` | str | Type of decay schedule ('cosine', 'linear', 'exponential') |
| `final_div_factor` | float, optional | If provided, overrides min_lr to be max_lr/final_div_factor |
| `verbose` | bool, optional | Whether to print validation messages |

### Returns

A function that calculates the learning rate for a given step.

### Examples

#### Cosine Decay (Default)

```python
from tlama_core.training import get_lr_scheduler

# Create a scheduler with cosine decay
scheduler = get_lr_scheduler(
    max_lr=1e-4,
    min_lr=1e-6,
    warmup_steps=1000,
    max_steps=50000,
    decay_type='cosine'
)

# Get learning rate at step 5000
lr = scheduler(5000)
```

#### Linear Decay

```python
# Create a scheduler with linear decay
scheduler = get_lr_scheduler(
    max_lr=1e-4,
    min_lr=1e-6,
    warmup_steps=1000,
    max_steps=50000,
    decay_type='linear'
)
```

#### Exponential Decay

```python
# Create a scheduler with exponential decay
scheduler = get_lr_scheduler(
    max_lr=1e-4,
    min_lr=1e-6,
    warmup_steps=1000,
    max_steps=50000,
    decay_type='exponential'
)
```

#### Using `final_div_factor`

```python
# Create a scheduler where min_lr is max_lr/100
scheduler = get_lr_scheduler(
    max_lr=1e-4,
    min_lr=None,  # Will be calculated
    warmup_steps=1000,
    max_steps=50000,
    final_div_factor=100
)
```

---

## get_combined_lr_scheduler

The `get_combined_lr_scheduler` function allows you to combine multiple schedulers over different step ranges.

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `schedulers` | Dict[Tuple[int, int], Callable] | Dictionary mapping (start_step, end_step) to scheduler functions |

### Returns

A function that calculates the learning rate based on the appropriate scheduler for the current step.

### Example

```python
from tlama_core.training import get_lr_scheduler, get_combined_lr_scheduler

# Define schedulers for different ranges
schedulers = {
    # First 10000 steps: Warmup and initial training
    (0, 10000): get_lr_scheduler(
        max_lr=1e-4, 
        min_lr=1e-5, 
        warmup_steps=1000, 
        max_steps=10000
    ),
    
    # Next 20000 steps: Fine-tuning with lower rate
    (10000, 30000): get_lr_scheduler(
        max_lr=1e-5, 
        min_lr=1e-6, 
        warmup_steps=0,  # No warmup for second phase 
        max_steps=20000
    )
}

# Create combined scheduler
combined_scheduler = get_combined_lr_scheduler(schedulers)

# Use in training loop
for step in range(30000):
    lr = combined_scheduler(step)
    # Update optimizer's learning rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
```

---

## get_multi_step_lr_scheduler

The `get_multi_step_lr_scheduler` function creates a scheduler that reduces the learning rate at specified milestones during training, based on the percentage of tokens processed.

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `max_lr` | float | Maximum learning rate after warmup |
| `warmup_steps` | int | Number of steps for linear warmup |
| `total_tokens` | int | Total number of training tokens |
| `tokens_per_step` | int | Number of tokens processed per step |
| `step_milestones` | List[float] | List of milestones as fractions of total_tokens (e.g., [0.5, 0.8, 0.9]) |
| `step_factors` | List[float] | List of learning rate factors at each milestone (e.g., [0.5, 0.1, 0.01]) |
| `min_lr` | float, optional | Minimum learning rate |
| `verbose` | bool, optional | Whether to print validation messages |

### Returns

A function that calculates the learning rate for a given step.

### Example

```python
from tlama_core.training import get_multi_step_lr_scheduler

# Create a scheduler with three learning rate drops
scheduler = get_multi_step_lr_scheduler(
    max_lr=1e-4,
    warmup_steps=2000,
    total_tokens=300_000_000_000,  # 300B tokens
    tokens_per_step=2048 * 2048,    # batch_size * seq_len
    step_milestones=[0.5, 0.8, 0.9],  # Drop LR at 50%, 80%, and 90% of training
    step_factors=[0.5, 0.1, 0.01]     # LR factors at each milestone
)

# Use in training loop
for step in range(total_steps):
    lr = scheduler(step)
    # Update optimizer's learning rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
```

---

## plot_lr_schedule

The `plot_lr_schedule` function helps visualize any learning rate schedule to confirm it behaves as expected.

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `lr_scheduler` | Callable | Learning rate scheduler function |
| `total_steps` | int | Total number of steps to plot |
| `title` | str, optional | Title for the plot |
| `figsize` | Tuple[int, int], optional | Figure size as (width, height) |
| `point_interval` | int, optional | Interval for showing points on the curve |

### Example

```python
from tlama_core.training import get_lr_scheduler, plot_lr_schedule

# Create a scheduler
scheduler = get_lr_scheduler(
    max_lr=1e-4,
    min_lr=1e-6,
    warmup_steps=1000,
    max_steps=50000
)

# Visualize the schedule
plot_lr_schedule(
    lr_scheduler=scheduler,
    total_steps=50000,
    title="Cosine LR Schedule with Warmup",
    figsize=(12, 6)
)
```

## Integration with Training Loop

Here's an example of how to use these schedulers in a PyTorch training loop:

```python
from tlama_core.training import get_lr_scheduler
import torch

# Define model, optimizer, loss function, etc.
model = YourModel()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0)  # Initial LR doesn't matter
dataloader = YourDataLoader()

# Create scheduler
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
    
    # Forward pass
    outputs = model(batch)
    loss = compute_loss(outputs, batch)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Optionally track and log the learning rate
    if step % 100 == 0:
        print(f"Step {step}, LR: {lr}")
```

## Comparing Different Schedulers

You can easily compare different learning rate schedules to choose the most appropriate one for your task:

```python
import matplotlib.pyplot as plt
from tlama_core.training import get_lr_scheduler

# Define step range
steps = range(10000)

# Create schedulers with different decay types
schedulers = {
    'cosine': get_lr_scheduler(1e-4, 1e-6, 1000, 10000, decay_type='cosine'),
    'linear': get_lr_scheduler(1e-4, 1e-6, 1000, 10000, decay_type='linear'),
    'exponential': get_lr_scheduler(1e-4, 1e-6, 1000, 10000, decay_type='exponential')
}

# Plot all schedules
plt.figure(figsize=(12, 6))
for name, scheduler in schedulers.items():
    lrs = [scheduler(step) for step in steps]
    plt.plot(steps, lrs, label=name)

plt.title('Comparison of LR Schedules')
plt.xlabel('Step')
plt.ylabel('Learning Rate')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()
```