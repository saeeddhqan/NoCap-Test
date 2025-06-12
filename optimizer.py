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

    def step(self, closure=None):
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
                weight_decay = group['weight_decay']
                lr = group['lr']
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['m'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['v'] = torch.zeros_like(p.data)
                    # state['v'] = torch.tensor(0.0, device=p.device)

                if weight_decay != 0.0:
                    p.data.mul_(1 - lr * weight_decay)

                m, v = state['m'], state['v']

                beta1, beta2 = group['betas']

                state['step'] += 1

                # Decay the first and second moment running average coefficient
                m.mul_(beta1).add_(p.data, alpha=1 - beta1)
                m_hat = m / (1 - beta1 ** state['step'])

                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                v_hat = v / (1 - beta2 ** state['step'])
                denom = v.sqrt().add_(group['eps'])

                # v.mul_(beta2).add_(grad.square().mean(), alpha=1 - beta2)
                # v_hat = v / (1 - beta2 ** state['step'])
                # denom = v.sqrt().add_(group['eps']) # one scalar

                # p.data.addcdiv_(m_hat, denom, value=-lr)
                p.data = m_hat.addcdiv_(grad, denom, value=-lr)

        return loss
