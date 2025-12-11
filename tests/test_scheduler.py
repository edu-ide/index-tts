import torch
import torch.optim as optim
import sys
import os
import matplotlib.pyplot as plt

# Add root to path to import utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from indextts.utils.scheduler import get_wsd_schedule_with_warmup
except ImportError:
    print("Could not import scheduler. Trying local import.")
    try:
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../indextts/utils")))
        from scheduler import get_wsd_schedule_with_warmup
    except ImportError:
        print("Failed to import scheduler.")
        sys.exit(1)

def test_wsd_scheduler():
    print("Testing WSD Scheduler...")
    
    model = torch.nn.Linear(10, 1)
    optimizer = optim.AdamW(model.parameters(), lr=1.0) # Base LR 1.0 for easy checking
    
    num_warmup_steps = 10
    num_training_steps = 100
    stable_ratio = 0.5 # Stable for 50% of remaining steps (approx 45 steps)
    
    # Expected timeline:
    # 0-10: Warmup (0.0 -> 1.0)
    # 10-55: Stable (1.0)
    # 55-100: Decay (1.0 -> 0.0)
    
    scheduler = get_wsd_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        stable_ratio=stable_ratio,
        min_lr_ratio=0.0,
        decay_style="linear"
    )
    
    lrs = []
    for step in range(num_training_steps):
        lr = scheduler.get_last_lr()[0]
        lrs.append(lr)
        scheduler.step()
        
    # Check Warmup
    print(f"Step 5 LR: {lrs[5]:.4f} (Expected ~0.5)")
    if not (0.4 < lrs[5] < 0.6):
        print("Warmup check failed!")
        return False
        
    # Check Stable
    print(f"Step 30 LR: {lrs[30]:.4f} (Expected 1.0)")
    if not (0.99 < lrs[30] < 1.01):
        print("Stable check failed!")
        return False
        
    # Check Decay
    print(f"Step 80 LR: {lrs[80]:.4f} (Expected < 1.0)")
    if not (lrs[80] < 0.9):
        print("Decay check failed!")
        return False
        
    print("Test Passed: WSD Scheduler behavior confirmed.")
    return True

if __name__ == "__main__":
    if test_wsd_scheduler():
        sys.exit(0)
    else:
        sys.exit(1)
