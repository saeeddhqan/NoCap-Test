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
                if len(state) == 0:
                    state['step'] = 0
                    state['m'] = torch.zeros_like(p.data)
                    state['v'] = torch.zeros_like(p.data)

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


class ThirdMomentum(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, total_steps=20000, ema_beta=0.9):
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, total_steps=total_steps, ema_beta=ema_beta)
        super(ThirdMomentum, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(ThirdMomentum, self).__setstate__(state)

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
                eps = group['eps']
                beta1, beta2 = group['betas']
                total_steps = group['total_steps']
                ema_beta = group['ema_beta']

                if len(state) == 0:
                    state['step'] = 0
                    state['m'] = torch.zeros_like(p.data)
                    state['v'] = torch.zeros_like(p.data)
                    state['thr_ema'] = torch.zeros((), device=p.device)

                state['step'] += 1
                step = state['step']
                if weight_decay != 0.0:
                    p.data.mul_(1 - lr * weight_decay)

                m, v, thr_ema = state['m'], state['v'], state['thr_ema']

                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                m_hat = m / (1 - beta1 ** step)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                v_hat = v / (1 - beta2 ** step)
                m_hat = m_hat / v_hat.sqrt().add_(eps)

                thr_ema.mul_(ema_beta).add_(m_hat.abs().mean().detach(), alpha=1 - ema_beta)

                mask = m_hat.abs() >= thr_ema
                if mask.any():
                    p.data[mask] -= lr * (m_hat[mask])

        return loss


class AdamwPrime(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, total_steps=20000, ema_beta=0.9):
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, total_steps=total_steps, ema_beta=ema_beta)
        super(AdamwPrime, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(AdamwPrime, self).__setstate__(state)

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
                eps = group['eps']
                beta1, beta2 = group['betas']
                total_steps = group['total_steps']
                ema_beta = group['ema_beta']

                if len(state) == 0:
                    state['step'] = 0
                    state['m'] = torch.zeros_like(p.data)
                    state['v'] = torch.zeros_like(p.data)
                    state['thr_ema'] = torch.zeros((), device=p.device)

                state['step'] += 1
                step = state['step']
                if weight_decay != 0.0:
                    p.data.mul_(1 - lr * weight_decay)

                m, v, thr_ema = state['m'], state['v'], state['thr_ema']

                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                m_hat = m / (1 - beta1 ** step)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                v_hat = v / (1 - beta2 ** step)
                m_hat = m_hat / (v_hat ** 0.25).add_(eps)

                thr_ema.mul_(ema_beta).add_(m_hat.abs().mean().detach(), alpha=1 - ema_beta)

                mask = m_hat.abs() >= thr_ema
                if mask.any():
                    p.data[mask] -= lr * (m_hat[mask])

        return loss