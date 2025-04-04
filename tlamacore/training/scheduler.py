import math
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Union, Optional, Tuple, Dict, Any, List
from rich.console import Console

console = Console()


def get_lr_scheduler(
        max_lr: float, 
        min_lr: float, 
        warmup_steps: int, 
        max_steps: int, 
        decay_type: str = 'cosine',
        final_div_factor: Optional[float] = None,
        verbose: bool = False
    ) -> Callable[[int], float]:
    """
    Create a learning rate scheduler with warmup and decay.
    
    Args:
        max_lr: Maximum learning rate after warmup
        min_lr: Minimum learning rate at the end of training
        warmup_steps: Number of steps for linear warmup
        max_steps: Total number of training steps
        decay_type: Type of decay schedule ('cosine', 'linear', 'exponential')
        final_div_factor: If provided, overrides min_lr to be max_lr / final_div_factor
        verbose: Whether to print validation messages
        
    Returns:
        Function that maps step number to learning rate
    
    Raises:
        ValueError: If any input parameter is invalid
    """
    # Validate input parameters
    if max_lr <= 0:
        raise ValueError(f"max_lr must be positive, got {max_lr}")
    
    if min_lr < 0:
        raise ValueError(f"min_lr must be non-negative, got {min_lr}")
    
    if max_lr < min_lr:
        raise ValueError(f"max_lr ({max_lr}) must be greater than min_lr ({min_lr})")
    
    if warmup_steps < 0:
        raise ValueError(f"warmup_steps must be non-negative, got {warmup_steps}")
    
    if max_steps <= 0:
        raise ValueError(f"max_steps must be positive, got {max_steps}")
    
    if warmup_steps >= max_steps:
        raise ValueError(f"warmup_steps ({warmup_steps}) must be less than max_steps ({max_steps})")
    
    if decay_type not in ['cosine', 'linear', 'exponential']:
        raise ValueError(f"decay_type must be one of ['cosine', 'linear', 'exponential'], got {decay_type}")
    
    # Handle final_div_factor
    if final_div_factor is not None:
        if final_div_factor <= 1:
            raise ValueError(f"final_div_factor must be greater than 1, got {final_div_factor}")
        min_lr = max_lr / final_div_factor
        if verbose:
            print(f"Setting min_lr to {min_lr} based on final_div_factor of {final_div_factor}")
    
    def get_lr(step: int) -> float:
        """
        Calculate learning rate for a given step.
        
        Args:
            step: Current training step
            
        Returns:
            Learning rate for the step
        """
        # Convert step to float for safety in calculations
        step_f = float(step)
        
        # 1) Linear warmup for warmup_steps steps
        if step < warmup_steps:
            return max_lr * (step_f + 1) / float(warmup_steps)
        
        # 2) If step > max_steps, return min learning rate
        if step >= max_steps:
            return min_lr
            
        # 3) In between, use specified decay down to min learning rate
        decay_ratio = (step_f - warmup_steps) / float(max_steps - warmup_steps)
        
        # Safety check (should be guaranteed by the previous conditions)
        decay_ratio = max(0.0, min(1.0, decay_ratio))
        
        if decay_type == 'cosine':
            # Cosine decay (smooth transition from max_lr to min_lr)
            coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        elif decay_type == 'linear':
            # Linear decay from max_lr to min_lr
            coeff = 1.0 - decay_ratio
        elif decay_type == 'exponential':
            # Exponential decay
            coeff = math.exp(-decay_ratio * 3)  # The factor 3 controls decay speed
        else:
            # Should never get here due to validation, but just in case
            coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        
        return min_lr + coeff * (max_lr - min_lr)
        
    return get_lr


def plot_lr_schedule(
        lr_scheduler: Callable[[int], float],
        total_steps: int,
        title: str = "Learning Rate Schedule",
        figsize: Tuple[int, int] = (10, 6),
        point_interval: int = 50
    ) -> None:
    """
    Plot a learning rate schedule for visualization.
    
    Args:
        lr_scheduler: Learning rate scheduler function
        total_steps: Total number of steps to plot
        title: Title for the plot
        figsize: Figure size as (width, height)
        point_interval: Interval for showing points on the curve
    """
    steps = list(range(total_steps))
    lr_values = [lr_scheduler(step) for step in steps]
    
    plt.figure(figsize=figsize)
    plt.plot(steps, lr_values, '-', linewidth=2)
    
    # Add points at regular intervals
    points_steps = steps[::point_interval]
    points_lr = [lr_values[i] for i in range(0, len(lr_values), point_interval)]
    plt.plot(points_steps, points_lr, 'o', markersize=4)
    
    plt.title(title)
    plt.xlabel("Step")
    plt.ylabel("Learning Rate")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


def get_combined_lr_scheduler(
        schedulers: Dict[Tuple[int, int], Callable[[int], float]]
    ) -> Callable[[int], float]:
    """
    Combine multiple schedulers over different step ranges.
    
    Args:
        schedulers: Dictionary mapping (start_step, end_step) to scheduler functions
        
    Returns:
        Combined scheduler function
    
    Example:
        schedulers = {
            (0, 1000): get_cosine_lr_scheduler(0.001, 0.01, 100, 1000),
            (1000, 2000): get_cosine_lr_scheduler(0.01, 0.0001, 0, 1000)
        }
        lr_fn = get_combined_lr_scheduler(schedulers)
    """
    # Sort the ranges by start step
    sorted_ranges = sorted(schedulers.keys(), key=lambda x: x[0])
    
    # Validate that ranges don't overlap and cover all steps
    for i in range(len(sorted_ranges) - 1):
        curr_start, curr_end = sorted_ranges[i]
        next_start, next_end = sorted_ranges[i + 1]
        
        if curr_end > next_start:
            raise ValueError(f"Overlapping ranges: {sorted_ranges[i]} and {sorted_ranges[i+1]}")
        
        if curr_end < next_start:
            raise ValueError(f"Gap between ranges: {sorted_ranges[i]} and {sorted_ranges[i+1]}")
    
    def combined_lr(step: int) -> float:
        for (start, end), scheduler in schedulers.items():
            if start <= step < end:
                # Adjust the step for this scheduler
                adjusted_step = step - start
                return scheduler(adjusted_step)
        
        # If step is beyond all ranges, use the last scheduler's final value
        last_start, last_end = sorted_ranges[-1]
        return schedulers[(last_start, last_end)](last_end - last_start - 1)
    
    return combined_lr


def get_multi_step_lr_scheduler(
    max_lr: float,
    warmup_steps: int,
    total_tokens: int,
    tokens_per_step: int,
    step_milestones: List[float],
    step_factors: List[float],
    min_lr: Optional[float] = None,
    verbose: bool = False,
):
    """
    Create a multi-step learning rate scheduler with warmup for pretraining.

    Args:
        mx_lr (float): Maximum learning rate after warmup
        warmup_steps (int): Number of steps for linear warmup
        total_tokens (int): Total number of training tokens
        tokens_per_step (int): Number of tokens processed per step
        step_milestones (List[float]): List of milestones as fractions of total_tokens (e.g., [0.8, 0.9])
        step_factors (List[float]): list of learning rate factors for each milestone (e.g., [0.316, 0.1])
        min_lr (Optional[float], optional): Minimun learning rate. Defaults to None.
        verbose (bool, optional): Wether to print validation messages. Defaults to False.
    """
    
    if max_lr <= 0:
        raise ValueError(f"Tlama-Core: 'max_lr' must be positive, got {max_lr}")
    
    if warmup_steps < 0:
        raise ValueError(f"Tlama-Core: 'warmup_steps' must be non-negative, got {warmup_steps}")
    
    if total_tokens <= 0:
        raise ValueError(f"Tlama-Core: 'total_tokens' must be positive, got {total_tokens}")
    
    if len(step_milestones) != len(step_factors):
        raise ValueError(f"Tlama-Core: 'step_milestones' and 'step_factors' must have the same length")
    
    if not all(0 < m < 1 for m in step_milestones):
        raise ValueError(f"Tlama-Core: All 'step_milestones' must be between 0 and 1")
    
    if not all(0 < f <= 1 for f in step_factors):
        raise ValueError(f"Tlama-Core: All 'step_factors' must be between 0 and 1")
    
    # Calculate total steps
    total_steps = math.ceil(total_tokens / tokens_per_step)
    
    # Calculate the number of steps for each milestone
    milestone_steps = [math.floor(m * total_steps) for m in step_milestones]
    
    # Set minimum learning rate if not provided
    if min_lr is None:
        min_lr = max_lr * min(step_factors)
    elif min_lr < 0:
        raise ValueError(f"Tlama-Core: 'min_lr' must be non-negative, got {min_lr}")
    
    if verbose:
        console.print(f"[green]Total steps: {total_steps}[/green]")
        console.print(f"[green]Milestone steps: {milestone_steps}[/green]")
        console.print(f"[green]LR factors at milestones: {step_factors}[/green]")
        console.print(f"[green]Max LR: {max_lr}, Min LR: {min_lr}[/green]")
        
    def get_lr(step: int) -> float:
        """
        Calculate learning rate for a given step.

        Args:
            step (int): Current training step

        Returns:
            float: Learning rate for the step
        """
        
        if step < warmup_steps:
            return max_lr * (step + 1) / float(warmup_steps)
        
        # Find applicable milestone
        for i, milestone_step in enumerate(milestone_steps):
            if step < milestone_step:
                return max_lr
            if i < len(milestone_steps) - 1 and step < milestone_steps[i + 1]:
                return max_lr * step_factors[i]
        
        return max_lr * step_factors[-1]
    
    return get_lr
            
def implement_paper_scheduler(
    max_lr: float,
    warmup_steps: int = 2000,
    total_tokens: int = None,
    tokens_per_step: int = None,
    batch_size: int = None,
    seq_len: int = None,
    gradient_clip_val: float = 1.0,
    verbose: bool = True
) -> Tuple[Callable[[int], float], float]:
    """
    Implement the specific scheduler described in the paper:
    - Warmup for 2000 steps
    - Decrease to 31.6% of max_lr after 80% of tokens
    - Decrease to 10% of max_lr after 90% of tokens
    - Gradient clipping at 1.0
    
    Args:
        max_lr: Maximum learning rate
        warmup_steps: Number of warmup steps (default 2000 as per paper)
        total_tokens: Total number of tokens in the dataset
        tokens_per_step: Tokens processed per step (if not provided, calculated from batch_size and seq_len)
        batch_size: Batch size (used to calculate tokens_per_step if not provided directly)
        seq_len: Sequence length (used to calculate tokens_per_step if not provided directly)
        gradient_clip_val: Gradient clipping value (default 1.0 as per paper)
        verbose: Whether to print scheduler details
        
    Returns:
        Tuple of (learning rate scheduler function, gradient_clip_val)
    """
    # Calculate tokens_per_step if not provided
    if tokens_per_step is None:
        if batch_size is None or seq_len is None:
            raise ValueError("Either tokens_per_step or both batch_size and seq_len must be provided")
        tokens_per_step = batch_size * seq_len
    
    if total_tokens is None:
        raise ValueError("total_tokens must be provided")
    
    # Paper-specific milestones and factors
    milestones = [0.8, 0.9]  # At 80% and 90% of training
    factors = [0.316, 0.1]   # Reduce to 31.6% and then 10% of max_lr
    
    scheduler = get_multi_step_lr_scheduler(
        max_lr=max_lr,
        warmup_steps=warmup_steps,
        total_tokens=total_tokens,
        tokens_per_step=tokens_per_step,
        step_milestones=milestones,
        step_factors=factors,
        verbose=verbose
    )
    
    return scheduler, gradient_clip_val


def plot_paper_scheduler(
    scheduler: Callable[[int], float],
    total_steps: int,
    warmup_steps: int = 2000,
    milestone_steps: List[int] = None,
    figsize: Tuple[int, int] = (12, 6)
) -> None:
    """Plot the learning rate schedule described in the paper."""
    steps = list(range(total_steps))
    lr_values = [scheduler(step) for step in steps]
    
    plt.figure(figsize=figsize)
    plt.plot(steps, lr_values, '-', linewidth=2)
    
    # Mark warmup end
    plt.axvline(x=warmup_steps, color='r', linestyle='--', alpha=0.7, label='End of warmup')
    
    # Mark milestones
    if milestone_steps:
        for i, step in enumerate(milestone_steps):
            plt.axvline(x=step, color='g', linestyle='--', alpha=0.7, 
                       label=f'Milestone {i+1} ({int(step/total_steps*100)}%)')
    
    plt.title("Multi-Step Learning Rate Schedule")
    plt.xlabel("Training Step")
    plt.ylabel("Learning Rate")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig("paper_scheduler_plot.png")


# Example usage
if __name__ == "__main__":
    # Example parameters
    max_lr = 0.0005
    total_tokens = 300_000_000_000  # 300B tokens
    batch_size = 2048
    seq_len = 2048
    tokens_per_step = batch_size * seq_len
    
    # Create the scheduler from the paper description
    scheduler, grad_clip = implement_paper_scheduler(
        max_lr=max_lr,
        warmup_steps=2000,
        total_tokens=total_tokens,
        batch_size=batch_size,
        seq_len=seq_len,
        verbose=True
    )
    
    # Calculate total steps and milestone steps for plotting
    total_steps = math.ceil(total_tokens / tokens_per_step)
    milestone_steps = [
        math.floor(0.8 * total_steps),
        math.floor(0.9 * total_steps)
    ]
    
    # Plot a fraction of the schedule to see details better
    view_steps = min(total_steps, 100000)  # Show first 100K steps or all if less
    
    plot_paper_scheduler(
        scheduler=scheduler,
        total_steps=view_steps,
        warmup_steps=2000,
        milestone_steps=[m for m in milestone_steps if m < view_steps]
    )
    
    print(f"Gradient clipping value: {grad_clip}")
    
    # Let's also check the LR at specific points
    print(f"LR at step 0: {scheduler(0)}")
    print(f"LR at step 1000 (50% of warmup): {scheduler(1000)}")
    print(f"LR at step 2000 (end of warmup): {scheduler(2000)}")
    print(f"LR at step 10000: {scheduler(10000)}")
    print(f"LR at 80% milestone: {scheduler(milestone_steps[0])}")
    print(f"LR at 90% milestone: {scheduler(milestone_steps[1])}")
    print(f"LR at final step: {scheduler(total_steps-1)}")






#=============================================================================================================================================================

