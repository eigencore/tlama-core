import os
import torch
import numpy as np

def load_tokens(filename: str)-> torch.Tensor:
    """
    Load tokenized data from a numpy file.
    
    Args:
        filename (str): Path to the numpy file containing tokenized data.
        
    Returns:
        torch.Tensor: Tensor containing the tokenized data.
    """
    npt = np.load(filename)
    npt = npt.astype(np.int32) 
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt


class DataLoaderLite:
    """
    Simple data loader for tokenized text data.
    
    Handles loading data from multiple shards and provides batches 
    for training and validation.
    """
    def __init__(self, 
                batch_size: int,
                seq_len: int,
                process_rank: int, 
                num_processes: int,
                split: str, 
                data_root: str, 
                master_process: bool):
        """
        Initialize the data loader.
        
        Args:
            batch_size (int): Size of each batch.
            seq_len (int): Length of each sequence.
            process_rank (int): Rank of the current process.
            num_processes (int): Total number of processes.
            split (str): Data split ('train' or 'val').
            data_root (str): Root directory containing the data shards.
            master_process (bool): Flag to indicate if this is the master process.
        """
        self.B = batch_size
        self.T = seq_len
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.master_process = master_process
        assert split in {'train', 'val'}

        # Get the shard filenames
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split}"
        if master_process:
            print(f"found {len(shards)} shards for split {split}")
        self.reset()

    def reset(self):
        """Reset the data loader to the beginning of the first shard."""
        # Initialize state at shard zero
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        """
        Get the next batch of data.
        
        Returns:
            Tuple of (input_tokens, target_tokens)
        """
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        x = (buf[:-1]).view(B, T)  # inputs
        y = (buf[1:]).view(B, T)   # targets
        
        # Advance the position in the tensor
        self.current_position += B * T * self.num_processes
        
        # If loading the next batch would be out of bounds, advance to next shard
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = B * T * self.process_rank
            
        return x, y