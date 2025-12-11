import torch
from torch.optim.optimizer import Optimizer
import math

class MARS(Optimizer):
    """
    Implements MARS optimizer (MARS-AdamW variant).
    
    Based on "MARS: Unleashing the Power of Variance Reduction for Training Large Models" (2024).
    Combines Stochastic Recursive Momentum with AdamW-style preconditioned updates.
    
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay coefficient (default: 1e-2)
        gamma (float, optional): coefficient for gradient correction (default: 0.025)
        optimize_1d (bool, optional): whether to apply MARS to 1D parameters (default: False)
        lr_1d (float, optional): learning rate for 1D parameters (default: None, uses lr)
        betas_1d (Tuple[float, float], optional): betas for 1D parameters (default: None, uses betas)
        weight_decay_1d (float, optional): weight decay for 1D parameters (default: None, uses weight_decay)
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-2,
        gamma=0.025,
        optimize_1d=False,
        lr_1d=None,
        betas_1d=None,
        weight_decay_1d=None,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            gamma=gamma,
            optimize_1d=optimize_1d,
            lr_1d=lr_1d if lr_1d is not None else lr,
            betas_1d=betas_1d if betas_1d is not None else betas,
            weight_decay_1d=weight_decay_1d if weight_decay_1d is not None else weight_decay,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None, bs=None):
        """
        Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
            bs (int, optional): Batch size (unused in this implementation but kept for API compatibility).
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            gamma = group['gamma']
            optimize_1d = group['optimize_1d']

            for p in group['params']:
                if p.grad is None:
                    continue

                # Determine if this is a 1D parameter (e.g. LayerNorm, Bias)
                is_1d = p.ndim <= 1
                
                # Select hyperparameters based on parameter type
                if is_1d and not optimize_1d:
                    lr = group['lr_1d']
                    beta1, beta2 = group['betas_1d']
                    wd = group['weight_decay_1d']
                    use_mars = False
                else:
                    lr = group['lr']
                    beta1, beta2 = group['betas']
                    wd = group['weight_decay']
                    use_mars = True

                grad = p.grad
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Note: last_grad will be initialized to grad at the end of step (approx mode)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                state['step'] += 1

                # Apply Weight Decay
                if wd != 0:
                    p.mul_(1 - lr * wd)

                # MARS Variance Reduction: Stochastic Recursive Momentum
                # c_t = g_t + gamma * (g_t - g_{t-1})
                if use_mars:
                    # Get last_grad (first step: use zeros as default)
                    last_grad = state.get('last_grad', torch.zeros_like(p))

                    # Correction term
                    # c_t = g_t + gamma * (g_t - g_{t-1})
                    # c_t = (1 + gamma) * g_t - gamma * g_{t-1}
                    # We modify p.grad in-place to save memory: p.grad <- c_t
                    
                    # 1. p.grad = p.grad * (1 + gamma)
                    grad.mul_(1 + gamma)
                    
                    # 2. p.grad = p.grad - gamma * last_grad
                    grad.add_(last_grad, alpha=-gamma)
                    
                    # Now grad holds the corrected gradient
                    corrected_grad = grad
                else:
                    corrected_grad = grad

                # AdamW Update Logic using corrected_grad
                
                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(corrected_grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(corrected_grad, corrected_grad, value=1 - beta2)

                denom = exp_avg_sq.sqrt().add_(group['eps'])

                # Bias correction
                # Note: Some implementations of AdamW omit bias correction for efficiency or use a different schedule.
                # We stick to standard AdamW bias correction.
                step = state['step']
                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step
                step_size = lr * math.sqrt(bias_correction2) / bias_correction1

                p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss

    @torch.no_grad()
    def update_last_grad(self):
        """
        Updates the 'last_grad' state for all parameters.
        Should be called after optimizer.step() and optimizer.zero_grad().
        """
        for group in self.param_groups:
            # Only update last_grad if we are using MARS for this group/param
            # But simpler to just update for all, or check logic.
            # We need to match the logic in step().
            
            optimize_1d = group['optimize_1d']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                is_1d = p.ndim <= 1
                if is_1d and not optimize_1d:
                    continue # Skip 1D params if not optimizing with MARS
                
                state = self.state[p]
                if 'last_grad' not in state:
                    state['last_grad'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                
                # Store current gradient
                state['last_grad'].copy_(p.grad)
