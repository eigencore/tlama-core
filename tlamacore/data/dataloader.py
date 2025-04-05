import os
import torch
import numpy as np
from rich.console import Console

console = Console()


def load_tokens_from_npy(filename):
    """
    Load tokens from a numpy file and convert them to a PyTorch tensor.

    Args:
        filename (str): The path to the numpy file containing the tokens.

    Returns:
        torch.Tensor: A PyTorch tensor containing the tokens.
    
    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file cannot be loaded as a numpy array.
    """
    try:
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Token file not found: {filename}")
        
        npt = np.load(filename)
        npt = npt.astype(np.float32)
        ptt = torch.tensor(npt, dtype=torch.long)
        return ptt
    except Exception as e:
        console.print(f"[red]Error loading tokens from {filename}: {str(e)}[/red]")
        raise


class DataLoaderLite:
    """
    A lightweight data loader for loading tokenized data in shards.
    
    Args:
        data_dir (str): Directory containing the tokenized data files.
        batch_size (int): Number of samples per batch.
        seq_len (int): Length of each sequence.
        process_rank (int, optional): Rank of the current process in a distributed setting. Default: 0
        num_process (int, optional): Total number of processes in a distributed setting. Default: 1
        split (str, optional): Split type ('train', 'valid', 'test'). Default: 'train'
        shuffle (bool, optional): Whether to shuffle the data. Default: False
        verbose (bool, optional): Whether to print verbose output. Default: True
    """
    
    VALID_SPLITS = ['train', 'val']
    
    def __init__(self, data_dir, batch_size, seq_len, process_rank=0, num_process=1, 
                 split='train', shuffle=False, verbose=True):
        
        self.data_dir = os.path.abspath(data_dir)

        # Validate input parameters
        if not os.path.exists(self.data_dir) or not os.path.isdir(self.data_dir):
            raise ValueError(f"Data directory does not exist or is not a directory: {data_dir}")
        if batch_size <= 0:
            raise ValueError(f"Batch size must be positive, got {batch_size}")
        if seq_len <= 0:
            raise ValueError(f"Sequence length must be positive, got {seq_len}")
        if process_rank < 0 or process_rank >= num_process:
            raise ValueError(f"Process rank must be between 0 and {num_process-1}, got {process_rank}")
        if num_process <= 0:
            raise ValueError(f"Number of processes must be positive, got {num_process}")
        if split not in self.VALID_SPLITS:
            raise ValueError(f"Split must be one of {self.VALID_SPLITS}, got '{split}'")
        
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.process_rank = process_rank
        self.num_process = num_process
        self.split = split
        self.shuffle = shuffle
        self.verbose = verbose
        
        self.shards = []
        self.current_shard = 0
        self.current_pos = 0
        self.tokens = None
        self.epoch = 0

        # Load the data from the specified directory
        self._load_data()
        if verbose and len(self.shards) > 0:
            console.print(f"[green]Tlama-Core: INFO: Found {len(self.shards)} shards for split '{self.split}'[/green]")
        self.reset()
        
    def _load_data(self):
        """
        Prepare shards for the data loader.
        
        Raises:
            RuntimeError: If no shards are found for the specified split.
        """
        shards = os.listdir(self.data_dir)
        shards = [s for s in shards if self.split in s and s.endswith('.npy')]
        shards = sorted(shards)
        shards = [os.path.join(self.data_dir, s) for s in shards]
        
        # Validate that the shards exist
        valid_shards = []
        for shard in shards:
            if os.path.exists(shard) and os.path.isfile(shard):
                valid_shards.append(shard)
            elif self.verbose:
                console.print(f"[yellow]Warning: Shard file not found: {shard}[/yellow]")
        
        self.shards = valid_shards
        if len(self.shards) == 0:
            raise RuntimeError(f"No valid shards found for split '{self.split}' in {self.data_dir}")
    
    def reset(self):
        """
        Reset the data loader to the beginning of the dataset.
        This should be called at the start of each epoch.
        """
        self.current_shard = 0
        self.epoch += 1
        
        try:
            self.tokens = load_tokens_from_npy(self.shards[self.current_shard])
            self.current_pos = self.batch_size * self.seq_len * self.process_rank
            
            if self.verbose:
                console.print(f"[blue]Reset dataloader to beginning of epoch {self.epoch}[/blue]")
                console.print(f"[blue]Loaded shard {self.current_shard + 1}/{len(self.shards)}: {os.path.basename(self.shards[self.current_shard])}[/blue]")
                console.print(f"[blue]Shard size: {len(self.tokens)} tokens[/blue]")
        except Exception as e:
            console.print(f"[red]Error resetting dataloader: {str(e)}[/red]")
            raise
        
    def __len__(self):
        """
        Get an estimate of the total number of batches in the dataset.
        
        Returns:
            int: Estimated number of batches.
        """
        if not self.tokens:
            return 0
        
        # Estimate batches per shard based on current shard
        tokens_per_shard = len(self.tokens)
        batches_per_shard = tokens_per_shard // (self.batch_size * self.seq_len)
        return batches_per_shard * len(self.shards) // self.num_process
    
    def next_batch(self):
        """
        Get the next batch of data.
        
        Returns:
            tuple: A tuple containing:
                - torch.Tensor: Input batch of tokenized data.
                - torch.Tensor: Target batch of tokenized data.
        
        Raises:
            StopIteration: If no more data is available and all shards have been processed.
        """
        if self.tokens is None:
            self.reset()
        
        _bsz, _seq_len = self.batch_size, self.seq_len
        
        # Check if we have enough tokens in the current shard
        required_tokens = _bsz * _seq_len + 1
        
        if self.current_pos + required_tokens > len(self.tokens):
            # Move to the next shard
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            
            # If we've gone through all shards, signal the end of an epoch
            if self.current_shard == 0:
                if self.verbose:
                    console.print(f"[blue]End of epoch {self.epoch}[/blue]")
                raise StopIteration("End of epoch")
            
            if self.verbose:
                console.print(f"[blue]Loading shard {self.current_shard + 1}/{len(self.shards)}: {os.path.basename(self.shards[self.current_shard])}[/blue]")
            
            self.tokens = load_tokens_from_npy(self.shards[self.current_shard])
            self.current_pos = self.batch_size * self.seq_len * self.process_rank
        
        # Ensure we don't go out of bounds
        end_pos = min(self.current_pos + required_tokens, len(self.tokens))
        buf = self.tokens[self.current_pos:end_pos]
        
        # Handle case where we don't have enough tokens (should be rare due to earlier check)
        if len(buf) < required_tokens:
            padding = torch.zeros(required_tokens - len(buf), dtype=torch.long)
            buf = torch.cat([buf, padding])
            if self.verbose:
                console.print(f"[yellow]Warning: Had to pad batch with {len(padding)} tokens[/yellow]")
        
        # Reshape into input and target tensors
        x = buf[:-1].view(_bsz, _seq_len)
        y = buf[1:].view(_bsz, _seq_len)
        
        # Shuffle if needed
        if self.shuffle:
            idx = torch.randperm(_bsz)
            x = x[idx]
            y = y[idx]
        
        # Update position for next batch
        self.current_pos += _bsz * _seq_len * self.num_process
        
        if self.current_pos + (_bsz * _seq_len * self.num_process + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens_from_npy(self.shards[self.current_shard])
            self.current_pos = _bsz * _seq_len * self.process_rank
        
        return x, y
    
    def __iter__(self):
        """
        Create an iterator for the data loader.
        
        Returns:
            self: The data loader instance as an iterator.
        """
        self.reset()
        return self
    
    def __next__(self):
        """
        Get the next batch when iterating.
        
        Returns:
            tuple: The next batch from next_batch().
        
        Raises:
            StopIteration: When no more batches are available.
        """
        try:
            return self.next_batch()
        except StopIteration:
            self.reset()  # Reset for next epoch if needed
            raise


