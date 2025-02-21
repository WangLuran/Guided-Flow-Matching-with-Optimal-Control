import argparse
import random
from functools import partial

import numpy as np
from tqdm import tqdm
import torch

from qm9 import dataset
from qm9.utils import compute_mean_mad
from qm9.models import DistributionProperty
from consts import qm9_with_h
from guided_sample import oc_opt_du, oc_opt_du_adjoint, oc_opt_du_gd, dflow_opt, flowgrad_opt
from utils import get_flow_model, get_classifier, setup_generation, get_classifier_score
from metrics import analyze_stability_for_molecules

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--prop', type=str,
        choices=['alpha', 'gap', 'homo', 'lumo', 'mu', 'Cv'], default='alpha'
    )
    parser.add_argument(
        '--method', type=str,
        choices=['oc', 'oc-adjoint', 'oc-gd', 'dflow', 'flowgrad', 'none'], default='oc'
    )
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--n-step', type=int, default=50)
    parser.add_argument('--gamma', type=float, default=0.01)
    parser.add_argument('--max-step', type=int, default=5)
    parser.add_argument('--max-iter', type=int, default=5)
    parser.add_argument('--optim', type=str, default='sgd', choices=['sgd', 'lbfgs'])
    parser.add_argument('--lr', type=float, default=1.)
    parser.add_argument('--max-grad-norm', type=float, default=10.)
    parser.add_argument('--max-gen', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save-path', type=str, default='gen.pt')
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

    print('Guided generation args:')
    print(vars(args))
    print()

    # Load classifier
    classifier, cargs = get_classifier(args.prop, args.device)
    print('Classifier loaded with args:')
    print(vars(cargs))
    print()

    # Load generative model
    flow, nodes_dist, deq, margs = get_flow_model(args.device)
    print('Flow model loaded with args:')
    print(vars(margs))
    print()

    max_n_nodes = qm9_with_h['max_n_nodes']
    dataloaders = dataset.retrieve_dataloaders(margs)[0]
    prop_dist = DistributionProperty(dataloaders['train'], [args.prop])
    property_norms = compute_mean_mad(dataloaders, [args.prop], margs.dataset)
    prop_dist.set_normalizer(property_norms)
    mean, mad = property_norms[args.prop]['mean'], property_norms[args.prop]['mad']

    maes = []
    atm_stab, mol_stab = [], []
    molecules = {'one_hot': [], 'x': [], 'node_mask': [], 'prop': [], 'mae': []}


    def loss_fn(z, ref, node_mask, edge_mask):
        pred, x, h = get_classifier_score(z, flow, classifier, deq, node_mask, edge_mask, args.device)
        return (pred - ref).abs().sum()


    def batch_optimize(max_gen=10000):
        n_batch = max_gen // args.batch_size
        pbar = tqdm(range(n_batch))
        n_nan = 0
        for _ in pbar:
            node_mask, edge_mask, nodesxsample = setup_generation(
                args.batch_size, nodes_dist, max_n_nodes, args.device
            )
            flow.set_conditional_param(node_mask, edge_mask, context=None)
            ref = prop_dist.sample_batch(nodesxsample).to(args.device)
            cur_loss_fn = partial(loss_fn, ref=ref, node_mask=node_mask, edge_mask=edge_mask)
            z0 = flow.sample_combined_position_feature_noise(args.batch_size, max_n_nodes, node_mask)
            if args.method == 'oc':
                z1 = oc_opt_du(
                    flow, cur_loss_fn, z0, n_step=args.n_step, gamma=args.gamma,
                    max_step=args.max_step, max_iter=args.max_iter, optim=args.optim, lr=args.lr,
                    max_grad_norm=args.max_grad_norm, reverse_t=True, verbose=False
                )
            elif args.method == 'oc-adjoint':
                z1 = oc_opt_du_adjoint(
                    flow, cur_loss_fn, z0, n_step=args.n_step, beta=1 - args.gamma,
                    max_step=args.max_step, lr=args.lr,
                    max_grad_norm=args.max_grad_norm, reverse_t=True, verbose=False
                )
            elif args.method == 'oc-gd':
                z1 = oc_opt_du_gd(
                    flow, cur_loss_fn, z0, n_step=args.n_step, gamma=args.gamma,
                    max_step=args.max_step, lr=args.lr,
                    max_grad_norm=args.max_grad_norm, reverse_t=True, verbose=False
                )
            elif args.method == 'dflow':
                z1 = dflow_opt(
                    flow, cur_loss_fn, z0, n_step=args.n_step, max_step=args.max_step,
                    max_iter=args.max_iter, optim=args.optim, lr=args.lr, max_grad_norm=args.max_grad_norm,
                    reverse_t=True, return_x0=False, verbose=False
                )
            elif args.method == 'flowgrad':
                z1 = flowgrad_opt(
                    flow, cur_loss_fn, z0, n_step=args.n_step, max_step=args.max_step,
                    lr=args.lr, max_grad_norm=args.max_grad_norm,
                    reverse_t=True, verbose=False
                )
            else:
                with torch.no_grad():
                    z1 = flow.decode(z0, node_mask, edge_mask, context=None)[-1]
            with torch.no_grad():
                pred, x, one_hot = get_classifier_score(z1, flow, classifier, deq, node_mask, edge_mask, args.device)
                mae = (pred - ref).abs().mean().item() * mad
                if np.abs(mae) > 100 or np.isnan(mae):
                    n_nan += 1
                    print(f'NaN detected: {n_nan} times!')
                else:
                    maes.append(mae)
                    one_hot = one_hot.detach().cpu()
                    x = x.detach().cpu()
                    node_mask = node_mask.detach().cpu()
                    molecules['one_hot'].append(one_hot)
                    molecules['x'].append(x)
                    molecules['node_mask'].append(node_mask)
                    molecules['prop'].append(ref.view(-1).detach().cpu() * mad + mean)
                    molecules['mae'].append(mae)
                    stab_dict = analyze_stability_for_molecules(
                        {'one_hot': [one_hot], 'x': [x], 'node_mask': [node_mask]}
                    )
                    atm_stab.append(stab_dict['atm_stable'])
                    mol_stab.append(stab_dict['mol_stable'])

            if len(maes):
                pbar.set_description(
                    f'MAE: {np.mean(maes):.4f} | '
                    f'STD: {np.std(maes):.4f} | '
                    f'Atm. stab.: {sum(atm_stab) / len(atm_stab):.4f} | '
                    f'Mol. stab.: {sum(mol_stab) / len(mol_stab):.4f}'
                )


    try:
        batch_optimize(args.max_gen)
    except KeyboardInterrupt:
        print('Interrupted!')
        exit(0)
    finally:
        if len(maes):
            print(f'Optimization MAE: {sum(maes) / len(maes):.6f}')
            print(f'Optimization Atm. stab.: {sum(atm_stab) / len(atm_stab):.6f}')
            print(f'Optimization Mol. stab.: {sum(mol_stab) / len(mol_stab):.6f}')
            torch.save(molecules, args.save_path)
            print(f'Saved generated molecules to {args.save_path}')
