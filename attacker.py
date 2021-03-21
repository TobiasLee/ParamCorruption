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
