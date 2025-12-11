import math
import torch
from torch.optim.lr_scheduler import LambdaLR

def get_wsd_schedule_with_warmup(
    optimizer, 
    num_warmup_steps: int, 
    num_training_steps: int, 
    num_cycles: float = 0.5, 
    last_epoch: int = -1,
    min_lr_ratio: float = 0.0,
    stable_ratio: float = 0.9,
    decay_style: str = "cosine"
):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.
    
    Modified for WSD (Warmup-Stable-Decay):
    - Warmup: Linear increase from 0 to lr
    - Stable: Constant lr for a portion of training (defined by stable_ratio)
    - Decay: Decay from lr to min_lr * lr over the remaining steps
    
    Args:
        optimizer ([`~torch.optim.Optimizer`]):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (`int`):
            The number of steps for the warmup phase.
        num_training_steps (`int`):
            The total number of training steps.
        num_cycles (`float`, *optional*, defaults to 0.5):
            The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0
            following a half-cosine).
        last_epoch (`int`, *optional*, defaults to -1):
            The index of the last epoch when resuming training.
        min_lr_ratio (`float`, *optional*, defaults to 0.0):
            The minimum learning rate ratio (final_lr / initial_lr).
        stable_ratio (`float`, *optional*, defaults to 0.9):
            The ratio of steps (excluding warmup) to keep the learning rate stable. 
            Note: This ratio applies to the *remaining* steps after warmup? 
            Or total steps? Usually WSD is defined on total steps.
            Let's define: 
            - Warmup end = num_warmup_steps
            - Stable end = num_warmup_steps + (num_training_steps - num_warmup_steps) * stable_ratio
            - Decay starts after Stable end.
        decay_style (`str`, *optional*, defaults to "cosine"):
            The style of decay: "cosine" or "linear".

    Return:
        `torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step):
        # Warmup Phase
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        
        # Calculate remaining steps after warmup
        remaining_steps = num_training_steps - num_warmup_steps
        
        # Stable Phase
        # stable_steps = int(remaining_steps * stable_ratio) 
        # Actually, WSD usually means stable for X% of TOTAL training, but let's stick to the ratio of remaining for easier config relative to warmup.
        # Let's interpret stable_ratio as: fraction of (total - warmup) to be stable.
        stable_steps = int(remaining_steps * stable_ratio)
        stable_end_step = num_warmup_steps + stable_steps
        
        if current_step < stable_end_step:
            return 1.0
        
        # Decay Phase
        decay_steps = num_training_steps - stable_end_step
        progress = float(current_step - stable_end_step) / float(max(1, decay_steps))
        
        if decay_style == "linear":
            return max(min_lr_ratio, 1.0 - (1.0 - min_lr_ratio) * progress)
        elif decay_style == "cosine":
            return max(min_lr_ratio, min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress)))
        else:
            # Default to constant if unknown style (should not happen)
            return 1.0

    return LambdaLR(optimizer, lr_lambda, last_epoch)
