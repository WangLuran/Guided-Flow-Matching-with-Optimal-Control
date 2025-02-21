import torch


def dflow_optimization(z0, dynamic, N=10, L_N=None, max_iter=20, lr=1.0):
    cnt = 0
    shape = z0.shape
    device = z0.device

    def loss_fn(cur_z):
        eps = 1e-3
        z = cur_z
        for i in range(N):
            t = torch.ones(shape[0], device=device) * i / N * (1. - eps) + eps
            vf = dynamic(z, t * 999)
            z = z + vf / N
        return L_N(z)

    def closure():
        nonlocal cnt
        cnt += 1
        optimizer.zero_grad()
        loss = loss_fn(z0)
        loss.backward()
        print(f'Iter {cnt}: Loss {loss.item():.4f}')
        return loss

    z0.requires_grad_(True)
    optimizer = torch.optim.LBFGS([z0], max_iter=max_iter, lr=lr)
    optimizer.step(closure)
    return z0.detach()
