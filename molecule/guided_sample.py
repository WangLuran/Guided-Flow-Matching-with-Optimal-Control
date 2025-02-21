import torch
from torch.nn.utils import clip_grad_norm_


def oc_opt_du(
        model, cost_fn, x0, n_step=10, gamma=0.03, max_step=5, max_iter=20, optim='sgd',
        lr=1.0, max_grad_norm=5., reverse_t=False, return_du=False, verbose=False
):
    r"""
    Optimal control optimization using fixed du. Iteratively update du according to
    .. math::
        du \gets \gamma du - \alpha\nabla_x L(x_1)

    :param model: the reference model
    :param cost_fn: the cost function on x1
    :param x0: the initial noise
    :param n_step: number of integration steps
    :param gamma: weight decay for the reference model
    :param max_step: maximum number of optimization steps
    :param max_iter: maximum number of optimization iterations
    :param optim: optimizer type, 'sgd' or 'lbfgs'
    :param lr: learning rate
    :param max_grad_norm: maximum gradient norm
    :param reverse_t: whether to reverse the time steps
    :param return_du: whether to return the optimized du
    :param verbose: whether to print the intermediate results
    :return: optimized generation, and the optimized du if return_du is True
    """
    cnt = 0
    device = x0.device
    du = torch.zeros(n_step + 1, *x0.size(), device=device, dtype=torch.float)
    du.requires_grad_(True)

    def odeint(func, state0):
        state = state0
        dt = -1 / n_step if reverse_t else 1 / n_step
        for t in ts[:-1]:
            state = state + func(t, state) * dt
        return state

    def ode_fn(t, state_t):
        xt = state_t[..., :-1]
        du_t = du[int(t * n_step)]
        vf = model(t, xt) + du_t
        norm2 = du_t.square().sum(dim=-1, keepdim=True)
        return torch.cat([vf, norm2], dim=-1)

    def loss_fn():
        norm2 = torch.zeros(*x0.size()[:-1], 1, device=device, dtype=torch.float)
        new_x0 = x0 + du[0 if reverse_t else -1]
        state_0 = torch.cat([new_x0, norm2], dim=-1)
        state_1 = odeint(ode_fn, state_0)
        x1 = state_1[..., :-1]
        norm2 = state_1[..., -1]
        return cost_fn(x1) + gamma * norm2.abs().mean() / 2

    def closure():
        nonlocal cnt
        cnt += 1
        optimizer.zero_grad()
        loss = loss_fn()
        if torch.isnan(loss).any():
            return torch.tensor(1e5, device=device)
        loss.backward()
        clip_grad_norm_([du], max_grad_norm)
        if verbose:
            print(f'Iter {cnt}: Loss {loss.item():.4f}')
        return loss

    t_list = [1, 0] if reverse_t else [0, 1]
    ts = torch.linspace(*t_list, n_step + 1, device=device, dtype=torch.float)
    if optim == 'sgd':
        optimizer = torch.optim.SGD([du], lr=lr)
    else:
        optimizer = torch.optim.LBFGS([du], max_iter=max_iter, lr=lr, line_search_fn='strong_wolfe')
    for step in range(max_step):
        loss = optimizer.step(closure)
        print(f'Step {step}: Loss {loss:.4f}')

    du = du.detach()
    with torch.no_grad():
        x1_opt = odeint(
            (lambda t, x: model(t, x) + du[int(t * n_step)]),
            x0 + du[0 if reverse_t else -1]
        ).detach()
    if return_du:
        return x1_opt, du
    return x1_opt


def oc_opt_du_adjoint(
        model, cost_fn, x0, n_step=50, max_step=5, lr=1.0, beta=0.99,
        max_grad_norm=5., reverse_t=False, return_du=False, verbose=False
):
    device = x0.device
    shape = x0.shape
    batch_size = shape[0]
    assert batch_size == 1, 'batch size must be 1 for flowgrad'

    u_ind = [i for i in range(n_step + 1)]
    du = []
    for _ in range(n_step + 1):
        du_i = torch.zeros(*x0.size(), device=device, dtype=torch.float)
        du_i.requires_grad_(True)
        du.append(du_i)

    optimizer = torch.optim.SGD(du, lr=lr)

    def ode_fn(t, xt):
        return model(t, xt) + du[int(t * n_step)]

    def get_traj(func, state_0):
        traj = torch.zeros(n_step + 1, *x0.size(), device=device, dtype=torch.float)
        cur_state = state_0.detach().clone()
        traj[0] = cur_state
        dt = -1 / n_step if reverse_t else 1 / n_step
        for i in range(n_step):
            t = ts[i]
            cur_state = cur_state + func(t, cur_state) * dt
            traj[i + 1] = cur_state
        return traj

    def odeint(func, state_0):
        state = state_0
        dt = -1 / n_step if reverse_t else 1 / n_step
        for t in ts[:-1]:
            state = state + func(t, state) * dt
        return state

    t_list = [1, 0] if reverse_t else [0, 1]
    ts = torch.linspace(*t_list, n_step + 1, device=device, dtype=torch.float)
    L_best = 1e6
    opt_u = []

    for step in range(max_step):
        optimizer.zero_grad()

        traj = get_traj(ode_fn, x0)
        inputs = torch.zeros(traj[-1].shape, device=device)
        inputs.data = traj[-1].to(device).detach().clone()
        inputs.requires_grad = True
        loss = cost_fn(inputs)
        lam = torch.autograd.grad(loss, inputs)[0]
        lam = lam.detach().clone()
        lam = clip_norm(lam, max_grad_norm)

        if loss.detach().cpu().numpy() < L_best:
            opt_u = []
            for ind in u_ind:
                opt_u.append(du[ind].detach().clone())
            L_best = loss.detach().cpu().numpy()

        lam_list = [lam]
        for j in range(n_step - 1, -1, -1):
            inputs = torch.zeros(lam.shape, device=device)
            inputs.data = traj[j].to(device).detach().clone()
            inputs.requires_grad = True
            t = ts[j]
            func = lambda x: (
                    x.contiguous().reshape(shape) + du[j].detach().clone() +
                    model(t.detach().clone(), x.contiguous().reshape(shape) + du[j].detach().clone())
            ).view(-1)
            output, vjp = torch.autograd.functional.vjp(func, inputs=inputs.view(-1), v=lam.detach().clone().view(-1))
            lam = vjp.detach().clone().contiguous().reshape(shape)
            lam = clip_norm(lam, max_grad_norm)
            lam_list.insert(0, lam)
            # lam_list.append(lam)
            del inputs

        for j in range(n_step):
            du[j] = du[j] * beta - lr * lam_list[j]
        print(f'Step {step}: Loss {loss:.4f}')

    del du
    du = opt_u
    with torch.no_grad():
        x1_opt = odeint(ode_fn, x0).detach()
    if return_du:
        return x1_opt, du
    return x1_opt


def oc_opt_du_gd(
        model, cost_fn, x0, n_step=10, gamma=0.03, max_step=5,
        lr=1.0, max_grad_norm=5., reverse_t=False, return_du=False, verbose=False
):
    cnt = 0
    device = x0.device
    x0.requires_grad_(True)
    du = torch.zeros(n_step + 1, *x0.size(), device=device, dtype=torch.float)
    du.requires_grad_(True)

    def odeint(func, state0):
        state = state0
        dt = -1 / n_step if reverse_t else 1 / n_step
        for t in ts[:-1]:
            state = state + func(t, state) * dt
        return state

    def ode_fn(t, xt):
        return model(t, xt) + du[int(t * n_step)]

    def loss_fn():
        return cost_fn(odeint(ode_fn, x0 + du[0 if reverse_t else -1]))

    def closure():
        nonlocal cnt
        cnt += 1
        optimizer.zero_grad()
        x0.grad = None
        loss = loss_fn()
        if torch.isnan(loss).any():
            return torch.tensor(1e5, device=device)
        loss.backward()
        clip_grad_norm_([du], max_grad_norm)
        if verbose:
            print(f'Iter {cnt}: Loss {loss.item():.4f}')

        # du.retain_grad()
        du.grad = du.grad + gamma * du.detach() / n_step
        return loss

    t_list = [1, 0] if reverse_t else [0, 1]
    ts = torch.linspace(*t_list, n_step + 1, device=device, dtype=torch.float)
    optimizer = torch.optim.SGD([du], lr=lr)
    for step in range(max_step):
        loss = optimizer.step(closure)
        print(f'Step {step}: Loss {loss:.4f}')

    du = du.detach()
    with torch.no_grad():
        x1_opt = odeint(ode_fn, x0 + du[0 if reverse_t else -1]).detach()
    if return_du:
        return x1_opt, du
    return x1_opt


def dflow_opt(
        model, cost_fn, x0, n_step=50, max_step=5, max_iter=5, optim='sgd', lr=1.0,
        max_grad_norm=5., reverse_t=False, return_x0=False, verbose=False
):
    r"""
    D-Flow optimization of the initial noise
    .. math::
        x_0 \gets x_0 - \nabla_{x_0} L(x_1)

    :param model: the reference model
    :param cost_fn: the cost function on x1
    :param x0: the initial noise
    :param n_step: number of integration steps
    :param max_step: maximum number of optimization steps
    :param max_iter: maximum number of optimization iterations
    :param optim: optimizer type, 'sgd' or 'lbfgs'
    :param lr: learning rate
    :param max_grad_norm: maximum gradient norm
    :param reverse_t: whether to reverse the time steps
    :param return_x0: whether to return the optimized x0
    :param verbose: whether to print the intermediate results
    :return: optimized generation, and the optimized x0 if return_x0 is True
    """
    cnt = 0

    def loss_fn(cur_x0):
        x = cur_x0
        tlist = [1, 0] if reverse_t else [0, 1]
        ts = torch.linspace(*tlist, n_step + 1, device=x.device)
        dt = -1 / n_step if reverse_t else 1 / n_step
        for t in ts[:-1]:
            # Euler step
            x = x + model(t, x) * dt
        return x.detach(), cost_fn(x)

    def closure():
        nonlocal cnt
        cnt += 1
        optimizer.zero_grad()
        _, loss = loss_fn(x0_opt)
        if torch.isnan(loss).any():
            return torch.tensor(1e5, device=x0.device)
        loss.backward()
        clip_grad_norm_(x0_opt, max_grad_norm)
        if verbose:
            print(f'Iter {cnt}: Loss {loss.item():.4f}')
        return loss

    x0_opt = x0.detach().clone()
    x0_opt.requires_grad_(True)
    if optim == 'sgd':
        optimizer = torch.optim.SGD([x0_opt], lr=lr)
    else:
        optimizer = torch.optim.LBFGS([x0_opt], max_iter=max_iter, lr=lr, line_search_fn='strong_wolfe')
    best_x0 = x0_opt.detach().clone()
    best_loss = 1e6
    for step in range(max_step):
        loss = optimizer.step(closure)
        if loss < best_loss:
            best_loss = loss
            best_x0 = x0_opt.detach().clone()
        print(f'Step {step}: Loss {loss:.4f}')

    x0_opt = best_x0.detach()
    with torch.no_grad():
        x1_opt, _ = loss_fn(x0_opt)
    if return_x0:
        return x1_opt, x0_opt
    return x1_opt


def clip_norm(x, max_norm):
    norm = x.square().sum()
    if norm > max_norm:
        x = x * (max_norm / (norm + 1e-6))
    return x


def flowgrad_opt(
        model, cost_fn, x0, n_step=50, max_step=5, lr=1.0,
        max_grad_norm=5., reverse_t=False, verbose=False
):
    device = x0.device
    shape = x0.shape
    batch_size = shape[0]
    assert batch_size == 1, 'batch size must be 1 for flowgrad'

    u_ind = [i for i in range(n_step + 1)]
    du = []
    for _ in range(n_step + 1):
        du_i = torch.zeros(*x0.size(), device=device, dtype=torch.float)
        du_i.requires_grad_(True)
        du.append(du_i)

    optimizer = torch.optim.SGD(du, lr=lr)

    def ode_fn(t, xt):
        return model(t, xt) + du[int(t * n_step)]

    def get_traj(func, state_0):
        traj = torch.zeros(n_step + 1, *x0.size(), device=device, dtype=torch.float)
        cur_state = state_0.detach().clone()
        traj[0] = cur_state
        dt = -1 / n_step if reverse_t else 1 / n_step
        for i in range(n_step):
            t = ts[i]
            cur_state = cur_state + func(t, cur_state) * dt
            traj[i + 1] = cur_state
        return traj

    def odeint(func, state_0):
        state = state_0
        dt = -1 / n_step if reverse_t else 1 / n_step
        for t in ts[:-1]:
            state = state + func(t, state) * dt
        return state

    t_list = [1, 0] if reverse_t else [0, 1]
    ts = torch.linspace(*t_list, n_step + 1, device=device, dtype=torch.float)
    L_best = 1e6
    opt_u = []

    for step in range(max_step):
        optimizer.zero_grad()

        traj = get_traj(ode_fn, x0)
        inputs = torch.zeros(traj[-1].shape, device=device)
        inputs.data = traj[-1].to(device).detach().clone()
        inputs.requires_grad = True
        loss = cost_fn(inputs)
        lam = torch.autograd.grad(loss, inputs)[0]
        lam = lam.detach().clone()
        lam = clip_norm(lam, max_grad_norm)

        if loss.detach().cpu().numpy() < L_best:
            opt_u = []
            for ind in u_ind:
                opt_u.append(du[ind].detach().clone())
            L_best = loss.detach().cpu().numpy()

        lam_list = [lam]
        for j in range(n_step - 1, -1, -1):
            inputs = torch.zeros(lam.shape, device=device)
            inputs.data = traj[j].to(device).detach().clone()
            inputs.requires_grad = True
            t = ts[j]
            func = lambda x: (
                    x.contiguous().reshape(shape) + du[j].detach().clone() +
                    model(t.detach().clone(), x.contiguous().reshape(shape) + du[j].detach().clone())
            ).view(-1)
            output, vjp = torch.autograd.functional.vjp(func, inputs=inputs.view(-1), v=lam.detach().clone().view(-1))
            lam = vjp.detach().clone().contiguous().reshape(shape)
            lam = clip_norm(lam, max_grad_norm)
            lam_list.insert(0, lam)
            # lam_list.append(lam)
            del inputs

        for j in range(n_step):
            du[j] = du[j] - lr * lam_list[j]
        print(f'Step {step}: Loss {loss:.4f}')

    del du
    du = opt_u
    with torch.no_grad():
        x1_opt = odeint(ode_fn, x0).detach()
    return x1_opt
