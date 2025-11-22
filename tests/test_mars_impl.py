import torch
import torch.nn as nn
import sys
import os

# Add root to path to import trainers
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from trainers.mars_optimizer import MARS
except ImportError:
    print("Could not import MARS from trainers.mars_optimizer. Trying local import.")
    try:
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../trainers")))
        from mars_optimizer import MARS
    except ImportError:
        print("Failed to import MARS.")
        sys.exit(1)

def test_mars_optimizer():
    print("Testing MARS Optimizer...")
    
    # Simple model
    model = nn.Linear(10, 1)
    optimizer = MARS(model.parameters(), lr=0.01, gamma=0.025, optimize_1d=True)
    
    # Dummy data
    inputs = torch.randn(5, 10)
    targets = torch.randn(5, 1)
    
    criterion = nn.MSELoss()
    
    initial_loss = criterion(model(inputs), targets).item()
    print(f"Initial Loss: {initial_loss:.4f}")
    
    for step in range(5):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        
        # Check if last_grad is present (should be None or 0 for first step before update)
        # Actually update_last_grad is called AFTER step.
        
        optimizer.step()
        optimizer.update_last_grad()
        
        print(f"Step {step+1}, Loss: {loss.item():.4f}")
        
        # Verify last_grad exists in state
        for group in optimizer.param_groups:
            for p in group['params']:
                state = optimizer.state[p]
                if 'last_grad' not in state:
                    print("Error: last_grad not found in state!")
                    return False
                if step > 0:
                     # last_grad should be non-zero usually
                    if torch.all(state['last_grad'] == 0):
                        print("Warning: last_grad is all zeros (could be chance, but unlikely)")

    final_loss = criterion(model(inputs), targets).item()
    print(f"Final Loss: {final_loss:.4f}")
    
    if final_loss < initial_loss:
        print("Test Passed: Loss decreased.")
        return True
    else:
        print("Test Failed: Loss did not decrease.")
        return False

if __name__ == "__main__":
    if test_mars_optimizer():
        sys.exit(0)
    else:
        sys.exit(1)
