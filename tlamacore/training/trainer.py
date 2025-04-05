# tlama/training/trainer.py
import os
import time
import torch
import numpy as np
from typing import Dict, Any, Optional, List, Callable, Union, Tuple
from rich.console import Console

console = Console()

class Trainer:
    """
    Trainer loop to train Tlama models.
    This class is responsible for managing:
    - Training and validation loops
    - Gradient accumulation
    - Logging
    - Checkpointing
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        train_data_loader: Any,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Callable] = None,
        val_data_loader: Optional[Any] = None,
        total_batch_size: int = 524288,  # ~0.5M tokens
        gradient_accumulation: bool = True,
        weight_decay: float = 0.1,
        learning_rate: float = 6e-4,
        epochs: int = 1,
        steps: Optional[int] = None,
        validation_steps: int = 250,
        checkpoint_steps: int = 5000,
        log_steps: int = 10,
        gradient_clip_val: float = 1.0,
        use_mixed_precision: bool = True,
        checkpoints_dir: str = "checkpoints",
        logs_dir: str = "logs",
        seed: int = 1337,
        verbose: bool = True,
        callbacks: List[Any] = None,
        master_process: bool = True,  # Ready for DDP later
    ):
        """
        Initialize the Trainer.
        
        Args:
            model: The model to train
            train_data_loader: DataLoader for training data
            optimizer: Optimizer to use (if None, will be created from model)
            scheduler: Learning rate scheduler (function that takes step and returns lr)
            val_data_loader: DataLoader for validation data
            total_batch_size: Total batch size in tokens
            gradient_accumulation: Whether to use gradient accumulation
            weight_decay: Weight decay for optimizer
            learning_rate: Learning rate (if scheduler is None)
            epochs: Number of epochs to train
            steps: Number of steps per epoch (if None, determined by data_loader)
            validation_steps: Run validation every N steps
            checkpoint_steps: Save checkpoint every N steps
            log_steps: Log metrics every N steps
            gradient_clip_val: Max norm for gradient clipping
            use_mixed_precision: Whether to use mixed precision training
            checkpoints_dir: Directory to save checkpoints
            logs_dir: Directory to save logs
            seed: Random seed
            verbose: Whether to print training information
            callbacks: List of callbacks to use
            master_process: Whether this is the master process (for DDP)
        """
        self.model = model
        self.train_data_loader = train_data_loader
        self.val_data_loader = val_data_loader
        self.scheduler = scheduler
        self.epochs = epochs
        self.steps = steps
        self.validation_steps = validation_steps
        self.checkpoint_steps = checkpoint_steps
        self.log_steps = log_steps
        self.gradient_clip_val = gradient_clip_val
        self.use_mixed_precision = use_mixed_precision
        self.checkpoints_dir = checkpoints_dir
        self.logs_dir = logs_dir
        self.verbose = verbose
        self.callbacks = callbacks or []
        self.master_process = master_process
        
        # Set up device
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.device_type = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = torch.device("mps")
            self.device_type = "cpu"  # MPS still uses CPU for some operations
        else:
            self.device = torch.device("cpu")
            self.device_type = "cpu"
        
        # Set seed for reproducibility
        self._set_seed(seed)
        
        if verbose and self.master_process:
            console.print(f"[blue]Tlama-Core: INFO: Using device: {self.device}[/blue]")
        
        # Move model to device
        self.model = model.to(self.device)
        
        # Set up gradient accumulation
        batch_size = getattr(train_data_loader, 'batch_size', 0)
        seq_len = getattr(train_data_loader, 'seq_length', 0) or getattr(train_data_loader, 'seq_len', 0)
        
        if batch_size == 0 or seq_len == 0:
            raise ValueError("DataLoader must have batch_size and seq_length/seq_len attributes")
        
        # Validate total batch size
        tokens_per_micro_batch = batch_size * seq_len
        if total_batch_size % tokens_per_micro_batch != 0:
            raise ValueError(f"Total batch size {total_batch_size} must be divisible by "
                             f"micro batch size {tokens_per_micro_batch} (batch_size * seq_len)")
        
        # Calculate gradient accumulation steps
        self.grad_accum_steps = total_batch_size // tokens_per_micro_batch if gradient_accumulation else 1
        
        if verbose and self.master_process:
            if self.grad_accum_steps > 1:
                console.print(f"[blue]Tlama-Core: INFO: Gradient accumulation enabled with "
                              f"{self.grad_accum_steps} steps.[/blue]")
            else:
                console.print("[blue]Tlama-Core: INFO: Gradient accumulation is disabled.[/blue]")
        
        # Enable high precision matrix multiplication
        torch.set_float32_matmul_precision("high")
        
        # Create optimizer if not provided
        if optimizer is None:
            if verbose and self.master_process:
                console.print(f"[blue]Tlama-Core: INFO: Creating optimizer.[/blue]")
            
            if hasattr(model, 'configure_optimizers'):
                self.optimizer = model.configure_optimizers(
                    weight_decay=weight_decay,
                    learning_rate=learning_rate,
                    device_type=self.device_type,
                    master_process=self.master_process
                )
            else:
                # Create a default AdamW optimizer if model doesn't have configure_optimizers
                param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
                decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
                nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
                optim_groups = [
                    {'params': decay_params, 'weight_decay': weight_decay},
                    {'params': nodecay_params, 'weight_decay': 0.0}
                ]
                self.optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8)
        else:
            self.optimizer = optimizer
        
        # Initialize internal state
        self.current_epoch = 0
        self.current_step = 0
        self.best_val_loss = float('inf')
        self.training_start_time = None
        self.epoch_start_time = None
        self.step_start_time = None
        
        # Create directories
        if self.master_process:
            os.makedirs(self.checkpoints_dir, exist_ok=True)
            os.makedirs(self.logs_dir, exist_ok=True)
            
            # Initialize log file
            self.log_file = os.path.join(self.logs_dir, "training_log.txt")
            with open(self.log_file, "w") as f:
                f.write("epoch,step,time,train_loss,learning_rate,grad_norm,val_loss,tokens_per_sec\n")
        
        # Call callbacks for init end
        for callback in self.callbacks:
            if hasattr(callback, 'on_init_end'):
                callback.on_init_end(self)
    
    def _set_seed(self, seed: int):
        """
        Set the random seed for reproducibility.
        
        Args:
            seed (int): The seed value to set.
        """
        if self.verbose and self.master_process:
            console.print(f"[blue]Tlama-Core: INFO: Setting seed to {seed}[/blue]")
        
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
    def _log_metrics(self, metrics: Dict[str, Any]):
        """
        Log metrics to file.
        
        Args:
            metrics (Dict[str, Any]): Metrics to log.
        """
        if not self.master_process:
            return
        
        # Build CSV line
        line = f"{self.current_epoch},{self.current_step},{time.time()}"
        
        for key in ['train_loss', 'learning_rate', 'grad_norm', 'val_loss', 'tokens_per_sec']:
            if key in metrics:
                line += f",{metrics[key]}"
            else:
                line += ","
        
        # Write to log file
        with open(self.log_file, "a") as f:
            f.write(f"{line}\n")
    
    def _save_checkpoint(self, val_loss: Optional[float] = None, is_best: bool = False):
        """
        Save a model checkpoint.
        
        Args:
            val_loss (Optional[float]): Validation loss to include in checkpoint
            is_best (bool): Whether this is the best model so far
        """
        if not self.master_process:
            return
        
        # Create checkpoint
        checkpoint = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epoch': self.current_epoch,
            'step': self.current_step,
        }
        
        # Add config if available
        if hasattr(self.model, 'config'):
            checkpoint['config'] = self.model.config
        
        # Add validation loss if provided
        if val_loss is not None:
            checkpoint['val_loss'] = val_loss
        
        # Save regular checkpoint
        checkpoint_name = f"checkpoint_step_{self.current_step:08d}.pt"
        checkpoint_path = os.path.join(self.checkpoints_dir, checkpoint_name)
        
        try:
            torch.save(checkpoint, checkpoint_path)
            if self.verbose:
                console.print(f"[blue]Tlama-Core: INFO: Saved checkpoint to {checkpoint_path}[/blue]")
        except Exception as e:
            console.print(f"[red]Tlama-Core: ERROR: Failed to save checkpoint: {e}[/red]")
        
        # Save best checkpoint if requested
        if is_best:
            best_path = os.path.join(self.checkpoints_dir, "checkpoint_best.pt")
            try:
                torch.save(checkpoint, best_path)
                if self.verbose:
                    console.print(f"[blue]Tlama-Core: INFO: Saved best checkpoint to {best_path}[/blue]")
            except Exception as e:
                console.print(f"[red]Tlama-Core: ERROR: Failed to save best checkpoint: {e}[/red]")
        
        # Call callbacks
        for callback in self.callbacks:
            if hasattr(callback, 'on_checkpoint_end'):
                callback.on_checkpoint_end(self, checkpoint_path)
    
    def validate(self) -> Optional[float]:
        """
        Run validation on the model.
        
        Returns:
            Optional[float]: Validation loss or None if no validation data
        """
        if self.val_data_loader is None:
            return None
        
        # Call callbacks
        for callback in self.callbacks:
            if hasattr(callback, 'on_validate_start'):
                callback.on_validate_start(self)
        
        self.model.eval()
        try:
            self.val_data_loader.reset()
        except:
            pass  # Some dataloaders might not have reset method
            
        val_loss_accum = 0.0
        val_loss_steps = 20  # Limit validation to 20 batches for speed
        
        with torch.no_grad():
            for i in range(val_loss_steps):
                try:
                    # Get batch
                    if hasattr(self.val_data_loader, 'next_batch'):
                        x, y = self.val_data_loader.next_batch()
                    else:
                        try:
                            batch = next(iter(self.val_data_loader))
                            x, y = batch
                        except (StopIteration, TypeError):
                            break
                    
                    # Move to device
                    x, y = x.to(self.device), y.to(self.device)
                    
                    # Forward pass
                    if self.use_mixed_precision and self.device_type == "cuda":
                        with torch.autocast(device_type=self.device_type, dtype=torch.bfloat16):
                            logits, loss = self.model(x, start_pos=0, targets=y)
                    else:
                        logits, loss = self.model(x, start_pos=0, targets=y)
                    
                    # Accumulate loss
                    val_loss_accum += loss.detach()
                except Exception as e:
                    console.print(f"[red]Tlama-Core: ERROR in validation: {e}[/red]")
                    break
        
        # Calculate average
        val_loss = val_loss_accum / (i + 1) if i >= 0 else float('inf')
        val_loss_value = val_loss.item()
        
        # Return model to training mode
        self.model.train()
        
        # Log validation loss
        if self.verbose and self.master_process:
            console.print(f"[blue]Tlama-Core: INFO: Validation loss: {val_loss_value:.4f}[/blue]")
        
        # Call callbacks
        metrics = {"val_loss": val_loss_value}
        for callback in self.callbacks:
            if hasattr(callback, 'on_validate_end'):
                callback.on_validate_end(self, metrics)
        
        return val_loss_value
    
    def train_step(self) -> Dict[str, float]:
        """
        Run a single training step (with gradient accumulation).
        
        Returns:
            Dict[str, float]: Metrics from the training step
        """
        # Call callbacks
        for callback in self.callbacks:
            if hasattr(callback, 'on_step_start'):
                callback.on_step_start(self)
        
        self.step_start_time = time.time()
        
        # Reset gradients
        self.optimizer.zero_grad()
        
        # Accumulate gradients
        loss_accum = 0.0
        for micro_step in range(self.grad_accum_steps):
            try:
                # Get batch
                if hasattr(self.train_data_loader, 'next_batch'):
                    x, y = self.train_data_loader.next_batch()
                else:
                    try:
                        batch = next(iter(self.train_data_loader))
                        x, y = batch
                    except (StopIteration, TypeError):
                        # If iterator is exhausted, try to reset or recreate
                        if hasattr(self.train_data_loader, 'reset'):
                            self.train_data_loader.reset()
                            x, y = self.train_data_loader.next_batch()
                        else:
                            self.train_data_loader = iter(self.train_data_loader)
                            batch = next(self.train_data_loader)
                            x, y = batch
                
                # Move to device
                x, y = x.to(self.device), y.to(self.device)
                
                # Forward pass
                if self.use_mixed_precision and self.device_type == "cuda":
                    with torch.autocast(device_type=self.device_type, dtype=torch.bfloat16):
                        logits, loss = self.model(x, start_pos=0, targets=y)
                else:
                    logits, loss = self.model(x, start_pos=0, targets=y)
                
                # Scale loss for gradient accumulation
                loss = loss / self.grad_accum_steps
                
                # Backward pass
                for callback in self.callbacks:
                    if hasattr(callback, 'on_backward_start'):
                        callback.on_backward_start(self, loss)
                
                loss.backward()
                
                for callback in self.callbacks:
                    if hasattr(callback, 'on_backward_end'):
                        callback.on_backward_end(self)
                
                # Accumulate loss
                loss_accum += loss.detach()
            
            except Exception as e:
                console.print(f"[red]Tlama-Core: ERROR in training step: {e}[/red]")
                continue
        
        # Apply gradient clipping
        try:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                self.gradient_clip_val
            )
        except Exception as e:
            console.print(f"[red]Tlama-Core: ERROR in gradient clipping: {e}[/red]")
            grad_norm = torch.tensor(0.0)
        
        # Update learning rate if scheduler is provided
        current_lr = self.optimizer.param_groups[0]['lr']
        if self.scheduler is not None:
            try:
                new_lr = self.scheduler(self.current_step)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = new_lr
                current_lr = new_lr
            except Exception as e:
                console.print(f"[red]Tlama-Core: ERROR in scheduler: {e}[/red]")
        
        # Step optimizer
        for callback in self.callbacks:
            if hasattr(callback, 'on_optimizer_step_start'):
                callback.on_optimizer_step_start(self)
        
        self.optimizer.step()
        
        for callback in self.callbacks:
            if hasattr(callback, 'on_optimizer_step_end'):
                callback.on_optimizer_step_end(self)
        
        # Synchronize if CUDA
        if self.device_type == "cuda":
            torch.cuda.synchronize()
        
        # Calculate step time and throughput
        step_time = time.time() - self.step_start_time
        
        # Calculate tokens processed
        batch_size = getattr(self.train_data_loader, 'batch_size', 0)
        seq_len = getattr(self.train_data_loader, 'seq_length', 0) or getattr(self.train_data_loader, 'seq_len', 0)
        tokens_processed = batch_size * seq_len * self.grad_accum_steps
        tokens_per_sec = tokens_processed / step_time if step_time > 0 else 0
        
        # Collect metrics
        metrics = {
            'train_loss': loss_accum.item(),
            'learning_rate': current_lr,
            'grad_norm': grad_norm.item(),
            'step_time_ms': step_time * 1000,
            'tokens_per_sec': tokens_per_sec
        }
        
        # Call callbacks
        for callback in self.callbacks:
            if hasattr(callback, 'on_step_end'):
                callback.on_step_end(self, metrics)
        
        return metrics
    
    def train(self) -> Dict[str, Any]:
        """
        Run the full training loop.
        
        Returns:
            Dict[str, Any]: Final training metrics
        """
        # Call callbacks for train start
        for callback in self.callbacks:
            if hasattr(callback, 'on_train_start'):
                callback.on_train_start(self)
        
        self.training_start_time = time.time()
        
        try:
            # Determine number of steps if not provided
            if self.steps is None:
                if hasattr(self.train_data_loader, '__len__'):
                    self.steps = len(self.train_data_loader)
                else:
                    self.steps = 1000  # Default to 1000 steps per epoch
            
            # Set model to training mode
            self.model.train()
            
            # Main training loop
            for epoch in range(self.epochs):
                self.current_epoch = epoch
                self.epoch_start_time = time.time()
                
                # Call callbacks for epoch start
                for callback in self.callbacks:
                    if hasattr(callback, 'on_epoch_start'):
                        callback.on_epoch_start(self)
                
                if self.verbose and self.master_process:
                    console.print(f"[green]Tlama-Core: INFO: Starting epoch {epoch+1}/{self.epochs}[/green]")
                
                for step in range(self.steps):
                    self.current_step = epoch * self.steps + step
                    
                    # Train step
                    metrics = self.train_step()
                    
                    # Log metrics
                    if self.current_step % self.log_steps == 0 or step == self.steps - 1:
                        if self.verbose and self.master_process:
                            console.print(
                                f"Step {self.current_step:5d} | "
                                f"loss: {metrics['train_loss']:.6f} | "
                                f"lr: {metrics['learning_rate']:.4e} | "
                                f"norm: {metrics['grad_norm']:.4f} | "
                                f"dt: {metrics['step_time_ms']:.2f}ms | "
                                f"tok/sec: {metrics['tokens_per_sec']:.2f}"
                            )
                        
                        self._log_metrics(metrics)
                    
                    # Validation
                    if self.current_step % self.validation_steps == 0 or step == self.steps - 1:
                        val_loss = self.validate()
                        
                        if val_loss is not None:
                            # Check if this is the best model
                            is_best = val_loss < self.best_val_loss
                            if is_best:
                                self.best_val_loss = val_loss
                            
                            # Save checkpoint
                            if self.current_step > 0 and (
                                self.current_step % self.checkpoint_steps == 0
                                or step == self.steps - 1
                                or is_best
                            ):
                                self._save_checkpoint(val_loss, is_best)
                    
                    # Regular checkpoint
                    elif self.current_step > 0 and self.current_step % self.checkpoint_steps == 0:
                        self._save_checkpoint()
                
                # End of epoch
                epoch_time = time.time() - self.epoch_start_time
                if self.verbose and self.master_process:
                    console.print(
                        f"[green]Tlama-Core: INFO: Epoch {epoch+1} completed "
                        f"in {epoch_time:.2f}s[/green]"
                    )
                
                # Call callbacks for epoch end
                for callback in self.callbacks:
                    if hasattr(callback, 'on_epoch_end'):
                        callback.on_epoch_end(self)
            
            # Save final checkpoint
            if self.master_process:
                final_val_loss = self.validate()
                self._save_checkpoint(final_val_loss)
                
                # Log final message
                total_time = time.time() - self.training_start_time
                console.print(
                    f"[green]Tlama-Core: INFO: Training completed in {total_time:.2f}s. "
                    f"Best validation loss: {self.best_val_loss:.6f}[/green]"
                )
            
        except KeyboardInterrupt:
            console.print("[yellow]Tlama-Core: INFO: Training interrupted by user[/yellow]")
            # Save emergency checkpoint
            if self.master_process:
                self._save_checkpoint(None, False)
        
        except Exception as e:
            console.print(f"[red]Tlama-Core: ERROR: Training failed with error: {e}[/red]")
            # Save emergency checkpoint
            if self.master_process:
                self._save_checkpoint(None, False)
            raise
        
        finally:
            # Call callbacks for train end
            for callback in self.callbacks:
                if hasattr(callback, 'on_train_end'):
                    callback.on_train_end(self)
        
        # Return final metrics
        return {
            'epochs_completed': self.current_epoch + 1,
            'steps_completed': self.current_step + 1,
            'best_val_loss': self.best_val_loss,
            'total_training_time': time.time() - self.training_start_time
        }