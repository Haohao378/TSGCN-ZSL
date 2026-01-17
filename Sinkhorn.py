import torch
import torch.nn as nn

class SinkhornDistance(nn.Module):

    def __init__(self, eps, max_iter, device, reduction='none'):
        super(SinkhornDistance, self).__init__()
        self.eps = eps
        self.max_iter = max_iter
        self.reduction = reduction
        self.device = device

    def forward(self, x, y):
        C = self._cost_matrix(x, y)
        x_points = x.shape[-2]
        y_points = y.shape[-2]
        if x.dim() == 2:
            batch_size = 1
        else:
            batch_size = x.shape[0]
        mu = torch.empty(batch_size, x_points, dtype=torch.float, requires_grad=False).fill_(1.0 / x_points).squeeze()
        nu = torch.empty(batch_size, y_points, dtype=torch.float, requires_grad=False).fill_(1.0 / y_points).squeeze()
        u = torch.zeros_like(mu)
        v = torch.zeros_like(nu)
        if self.device is not None:
            mu = mu.to(self.device)
            nu = nu.to(self.device)
            u = u.to(self.device)
            v = v.to(self.device)
            C = C.to(self.device)
        thresh = 0.1
        for i in range(self.max_iter):
            u1 = u
            u = self.eps * (torch.log(mu + 1e-08) - torch.logsumexp(self.M(C, u, v), dim=-1)) + u
            v = self.eps * (torch.log(nu + 1e-08) - torch.logsumexp(self.M(C, u, v).transpose(-2, -1), dim=-1)) + v
            err = (u - u1).abs().sum(-1).mean()
            if err.item() < thresh:
                break
        U, V = (u, v)
        pi = torch.exp(self.M(C, U, V))
        cost = torch.sum(pi * C, dim=(-2, -1))
        if self.reduction == 'mean':
            cost = cost.mean()
        elif self.reduction == 'sum':
            cost = cost.sum()
        return (cost, pi, C)

    def M(self, C, u, v):
        return (-C + u.unsqueeze(-1) + v.unsqueeze(-2)) / self.eps

    @staticmethod
    def _cost_matrix(x, y, p=2):
        x_col = x.unsqueeze(-2)
        y_lin = y.unsqueeze(-3)
        C = torch.sum(torch.abs(x_col - y_lin) ** p, -1)
        return C

    @staticmethod
    def ave(u, u1, tau):
        return tau * u + (1 - tau) * u1