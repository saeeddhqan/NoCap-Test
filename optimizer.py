# rmsprop_momentum.py
import math
import torch
from torch.optim.optimizer import Optimizer, required


class RMSpropMomentum(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay)
        super(RMSpropMomentum, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(RMSpropMomentum, self).__setstate__(state)

    def step(self, epoch, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('does not support sparse gradients')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    # state['exp_avg_sq'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.tensor(0.0, device=p.device)
                    # state['w_exp_avg'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                wexp_avg = state['w_exp_avg']

                beta1, beta2 = group['betas']

                state['step'] += 1

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                # wexp_avg.mul_(beta1).add_(p.data, alpha=1 - beta1)

                # exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                # denom = exp_avg_sq.sqrt().add_(group['eps'])

                exp_avg_sq.mul_(beta2).add_(grad.square().mean(), alpha=1 - beta2)
                denom = exp_avg_sq.sqrt().add_(group['eps'])

                # if epoch < 2:print(denom.mean())
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                # p.data.addcdiv_(-step_size, exp_avg, denom)
                p.data.add_(
                # wexp_avg.add_(
                    torch.mul(p.data, group['weight_decay']).addcdiv_(exp_avg, denom, value=1), alpha=-step_size)
                # p.data = wexp_avg

        return loss
