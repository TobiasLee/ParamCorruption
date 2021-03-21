import torch
from torch.optim import Optimizer


class RandomAttacker(Optimizer):
    def __init__(self, params, mean=0, var=1, eps=1e-4):
        defaults = dict(mean=mean, var=var, eps=eps)
        self.mean = mean
        self.var = var
        self.eps = eps
        super(RandomAttacker, self).__init__(params, defaults)

    def step(self, closure=None):
        for group in self.param_groups:
            for p in group['params']:
                random_corruption = self.eps * torch.normal(mean=self.mean, std=self.var, size=p.size())
                p.data.add_(random_corruption)


class GradAttacker(Optimizer):
    def __init__(self, params, lr=1e-4, eps=1e-3, N=0, LP='L2'):
        if LP.lower() not in ["l2", "linf"]:
            raise ValueError("Invalid LP: {}".format(LP))
        self.eps = eps
        self.LP = LP.lower()
        self.N = N
        defaults = dict(lr=lr)
        super(GradAttacker, self).__init__(params, defaults)

    def step(self, closure=None):
        # normalize
        if self.LP == 'l2':
            L = 0
            for group in self.param_groups:
                for i, p in enumerate(group['params']):
                    if p.grad is None:
                        continue
                    grad = p.grad.data
                    L = L + (grad ** 2).sum()
            L = L ** .5
            for group in self.param_groups:
                for i, p in enumerate(group['params']):
                    if p.grad is None:
                        continue
                    p.grad.data = p.grad.data / L
        elif self.LP == 'linf':
            for group in self.param_groups:
                for i, p in enumerate(group['params']):
                    if p.grad is None:
                        continue
                    grad = p.grad.data
                    p.grad.data = (grad > 0).float() - (grad < 0).float()

        # update attack
        for group in self.param_groups:
            for i, p in enumerate(group['params']):
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]
                if 'attack' not in state:
                    state['attack'] = torch.zeros_like(p.data)
                state['old_attack'] = torch.clone(state['attack']).detach()
                state['attack'].add_(grad, alpha=group['lr'])

        # project
        if self.N != 0:
            atks = []
            for group in self.param_groups:
                for i, p in enumerate(group['params']):
                    if p.grad is None:
                        continue
                    state = self.state[p]
                    atks.append(state['attack'].view(-1))
            all_atk = torch.cat(atks)
            topk_value, _ = all_atk.abs().topk(self.N)
            thr = topk_value[-1]
            for group in self.param_groups:
                for i, p in enumerate(group['params']):
                    if p.grad is None:
                        continue
                    state = self.state[p]
                    mask = state['attack'].abs() < thr
                    state['attack'].masked_fill_(mask, 0)
        L = 0
        if self.LP == 'l2':
            for group in self.param_groups:
                for i, p in enumerate(group['params']):
                    if p.grad is None:
                        continue
                    state = self.state[p]
                    L = L + (state['attack'] ** 2).sum()
            L = L ** .5
            if L > self.eps:
                for group in self.param_groups:
                    for i, p in enumerate(group['params']):
                        if p.grad is None:
                            continue
                        state = self.state[p]
                        state['attack'].mul_(self.eps / L)
        elif self.LP == 'linf':
            for group in self.param_groups:
                for i, p in enumerate(group['params']):
                    if p.grad is None:
                        continue
                    state = self.state[p]
                    state['attack'].clamp_(-self.eps, self.eps)

        # update params
        for group in self.param_groups:
            for i, p in enumerate(group['params']):
                if p.grad is None:
                    continue
                state = self.state[p]
                p.data.add_(state['attack'] - state['old_attack'])
