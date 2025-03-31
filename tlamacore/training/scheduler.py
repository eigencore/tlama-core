import math

def get_cosine_lr_scheduler(max_lr, min_lr, warmup_steps, max_steps):
    """
    Create a cosine learning rate scheduler with warmup.
    
    Args:
        max_lr: Maximum learning rate after warmup
        min_lr: Minimum learning rate at the end of training
        warmup_steps: Number of steps for linear warmup
        max_steps: Total number of training steps
        
    Returns:
        Function that maps step number to learning rate
    """
    def get_lr(step):
        # 1) Linear warmup for warmup_steps steps
        if step < warmup_steps:
            return max_lr * (step + 1) / warmup_steps
        
        # 2) If step > max_steps, return min learning rate
        if step > max_steps:
            return min_lr
            
        # 3) In between, use cosine decay down to min learning rate
        decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # Coeff starts at 1 and goes to 0
        return min_lr + coeff * (max_lr - min_lr)
        
    return get_lr