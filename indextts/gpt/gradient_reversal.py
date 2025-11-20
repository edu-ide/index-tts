"""
Gradient Reversal Layer (GRL) implementation for IndexTTS2 3-stage training.

Based on:
- Ganin et al. 2016, "Domain-adversarial training of neural networks"
  JMLR 17(59):1-35
- IndexTTS2 (arXiv:2506.21619v2) Stage 2 emotion disentanglement

The GRL performs identity transformation during forward pass,
but reverses gradients during backward pass to enable adversarial training.
"""

import torch
import torch.nn as nn
from torch.autograd import Function


class GradientReversalFunction(Function):
    """
    Gradient Reversal Layer function.

    Forward: output = input (identity)
    Backward: grad_input = -lambda * grad_output (sign reversal)
    """

    @staticmethod
    def forward(ctx, x, lambda_):
        """
        Forward pass: identity transformation.

        Args:
            x: Input tensor
            lambda_: Reversal strength (typically starts at 0, gradually increases)

        Returns:
            Same tensor as input
        """
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass: reverse gradient by multiplying with -lambda.

        Args:
            grad_output: Gradient from downstream layers

        Returns:
            Reversed gradient for input, None for lambda_
        """
        lambda_ = ctx.lambda_
        grad_input = -lambda_ * grad_output
        return grad_input, None


class GradientReversalLayer(nn.Module):
    """
    Gradient Reversal Layer module.

    Usage in IndexTTS2 Stage 2:
        # Create GRL
        grl = GradientReversalLayer(lambda_=1.0)

        # In forward pass
        emo_vec = self.emo_perceiver_encoder(...)
        emo_vec_reversed = grl(emo_vec)
        speaker_pred = self.speaker_classifier(emo_vec_reversed)

        # Loss
        tts_loss = compute_tts_loss(...)
        speaker_loss = compute_speaker_classification_loss(...)
        total_loss = tts_loss + speaker_loss  # GRL handles the minus sign

    Lambda scheduling (recommended):
        # Gradually increase lambda from 0 to 1 over warmup period
        p = current_step / total_steps
        lambda_ = 2.0 / (1.0 + exp(-10 * p)) - 1.0
    """

    def __init__(self, lambda_=1.0):
        """
        Initialize GRL.

        Args:
            lambda_: Initial reversal strength (default: 1.0)
                    - 0.0: no reversal (normal gradient flow)
                    - 1.0: full reversal (standard GRL)
                    - Can be dynamically adjusted during training
        """
        super(GradientReversalLayer, self).__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        """
        Apply gradient reversal.

        Args:
            x: Input tensor of any shape

        Returns:
            Same tensor (forward), reversed gradient (backward)
        """
        return GradientReversalFunction.apply(x, self.lambda_)

    def set_lambda(self, lambda_):
        """
        Update reversal strength.

        Args:
            lambda_: New reversal strength
        """
        self.lambda_ = lambda_


def get_lambda_schedule(current_step, total_steps, schedule='exponential'):
    """
    Get lambda value for current training step.

    Args:
        current_step: Current training step
        total_steps: Total training steps
        schedule: Scheduling strategy
            - 'constant': Always 1.0
            - 'linear': Linear from 0 to 1
            - 'exponential': Smooth curve (recommended by Ganin et al.)

    Returns:
        Lambda value for GRL

    Example:
        >>> for step in range(0, 10000, 1000):
        ...     lambda_ = get_lambda_schedule(step, 10000)
        ...     grl.set_lambda(lambda_)
    """
    if schedule == 'constant':
        return 1.0

    elif schedule == 'linear':
        return min(1.0, current_step / total_steps)

    elif schedule == 'exponential':
        # Formula from Ganin et al. 2016
        # Smooth S-curve from 0 to 1
        import math
        p = current_step / total_steps
        return 2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0

    else:
        raise ValueError(f"Unknown schedule: {schedule}")


# Example usage
if __name__ == "__main__":
    print("=" * 60)
    print("Gradient Reversal Layer Test")
    print("=" * 60)

    # Create GRL
    grl = GradientReversalLayer(lambda_=1.0)

    # Test forward pass
    x = torch.randn(4, 128, requires_grad=True)
    y = grl(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Forward pass: output == input? {torch.allclose(x, y)}")

    # Test backward pass
    loss = y.sum()
    loss.backward()

    print(f"\nBackward pass:")
    print(f"Expected gradient: all -1.0 (reversed from 1.0)")
    print(f"Actual gradient: {x.grad[0, :5]}")
    print(f"Gradient reversed? {torch.allclose(x.grad, -torch.ones_like(x))}")

    # Test lambda scheduling
    print(f"\n{'='*60}")
    print("Lambda Scheduling Test")
    print(f"{'='*60}")
    print(f"{'Step':<10} {'Exponential':<15} {'Linear':<15}")
    print("-" * 40)
    for step in [0, 1000, 5000, 10000, 20000, 30000]:
        exp_lambda = get_lambda_schedule(step, 30000, 'exponential')
        lin_lambda = get_lambda_schedule(step, 30000, 'linear')
        print(f"{step:<10} {exp_lambda:<15.4f} {lin_lambda:<15.4f}")
