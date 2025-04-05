# DataLoader

This module provides a lightweight loader for tokenized `.npy` files, returning batches suitable for training language models.

---

## Functions

### `load_tokens_from_npy(filename)`

```python
load_tokens_from_npy(filename: str) -> torch.Tensor
```

Loads tokens from a `.npy` file and converts them to a PyTorch tensor.

#### Arguments

- `filename`: Path to the file to load.

#### Returns

- A `torch.Tensor` with the loaded tokens.

#### Raises

- `FileNotFoundError` if the file does not exist.
- Other exceptions with error message if loading fails.

---

## Classes

### `DataLoaderLite`

A minimal data loader for language model training on tokenized datasets, supporting sharding and distributed settings.

#### Constructor

```python
DataLoaderLite(
    data_dir: str,
    batch_size: int,
    seq_len: int,
    process_rank: int = 0,
    num_process: int = 1,
    split: str = 'train',
    shuffle: bool = False,
    verbose: bool = True
)
```

Initializes the data loader and prepares the list of data shards.

##### Arguments

- `data_dir`: Path to directory containing `.npy` token files.
- `batch_size`: Number of samples per batch.
- `seq_len`: Sequence length for each sample.
- `process_rank`: Index of current process (for distributed use).
- `num_process`: Total number of processes.
- `split`: Which split to use (`'train'` or `'val'`).
- `shuffle`: Whether to shuffle batches.
- `verbose`: Print logs and warnings.

---

### `reset()`

```python
reset() -> None
```

Loads the first shard and resets internal counters. Should be called at the beginning of each epoch.

---

### `_load_data()` 

```python
_load_data() -> None
```

Scans the data directory and collects valid `.npy` shards that match the selected split.

---

### `__len__()`

```python
__len__() -> int
```

Returns an estimate of the total number of batches across all shards.

---

### `next_batch()`

```python
next_batch() -> Tuple[torch.Tensor, torch.Tensor]
```

Returns the next `(x, y)` pair of input/target batches. Moves to the next shard when needed.

#### Returns

- `x`: Tensor of shape `(batch_size, seq_len)`
- `y`: Tensor of shape `(batch_size, seq_len)`

#### Raises

- `StopIteration` when all shards have been processed (end of epoch).

---

### `__iter__()`

```python
__iter__() -> Self
```

Resets the loader and returns self for iteration.

---

### `__next__()`

```python
__next__() -> Tuple[torch.Tensor, torch.Tensor]
```

Alias for `next_batch()` in iterator mode. Automatically resets for the next epoch.