import argparse
import torch
import torch.nn as nn
from datetime import datetime
import gc
import pandas as pd
import sys, os
import random
from torch.utils.data import Subset
sys.stdout.flush()
sys.path.append('rlhf_fm/peptide')
print("CUDA device count:", torch.cuda.device_count())
from torch.utils.data import DataLoader, random_split
from models_con.utils import process_dic
from models_con.pep_dataloader import PepDataset
import scipy.linalg
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
os.environ["CUDA_VISIBLE_DEVICES"] = str(rank % 2)
print(f"Process {rank} is using GPU {os.environ['CUDA_VISIBLE_DEVICES']}")
import wandb
from tqdm import tqdm
from pepflow.utils.misc import BlackHole, inf_iterator, load_config, seed_all, get_logger, get_new_log_dir, current_milli_time
from pepflow.utils.data import PaddingCollate
from pepflow.utils.train import ScalarMetricAccumulator, count_parameters, get_optimizer, get_scheduler, log_losses, recursive_to, sum_weighted_losses
from models_con.flow_model import FlowModel , encode
from models_con.torsion import full_atom_reconstruction
from pepflow.modules.protein.constants import AA, BBHeavyAtom, max_num_heavyatoms
from torchcfm.conditional_flow_matching import *

from madrax.ForceField import ForceField
from madrax import utils, dataStructures
import shutil
from copy import deepcopy
import torch.nn.functional as F
import time
from torch.autograd.functional import jvp
from pepflow.modules.so3.dist import centered_gaussian,uniform_so3
from pepflow.modules.protein.constants import (AA, max_num_heavyatoms, max_num_hydrogens,
                        restype_to_heavyatom_names, 
                        restype_to_hydrogen_names,
                        BBHeavyAtom, non_standard_residue_substitutions)
from data import so3_utils
from models_con.sample import save_samples_sc, save_samples_sc_opt

def convert_to_custom_format(batch):
    formatted_data = []
    coords = []
    seqs_1 = batch['aa'] 
    chain_ids = batch['chain_nb'] 
    pos_heavyatom = batch['pos_heavyatom']

    for chain_index in range(seqs_1.shape[0]):
        for i, residue_seq in enumerate(seqs_1[chain_index]):
            residue_index = min(residue_seq.item(), 20)
            residue_enum = AA(residue_index) 
            residue_name = residue_enum.name 
            residue_number = i + 1 
            chain_id = chain_ids[chain_index, i].item()

            heavyatom_names = restype_to_heavyatom_names.get(residue_enum, [])

            for atom_index, atom_name in enumerate(heavyatom_names):
                if atom_name == '' or atom_index == 14:
                    continue

                atom_pos = pos_heavyatom[chain_index, i, atom_index]
                if atom_pos is None:
                    raise ValueError(f"{residue_name}_{residue_number}_{atom_name}_{chain_id} is none")

                formatted_str = f"{residue_name}_{residue_number}_{atom_name}_C{chain_id}_0_0"
                formatted_data.append(formatted_str)
                coords.append(atom_pos)

    coords_tensor = torch.stack(coords).unsqueeze(0) 

    formatted_data_batch = [formatted_data]

    return coords_tensor, formatted_data_batch

class Reward:
    def __init__(self):
        self.device =  'cpu'
        self.forceField_Obj = ForceField(device=self.device)

    def get_value(self, trans, rotmats, batch):
        with torch.no_grad():
            rotmats_1, trans_1, angles_1, seqs_1 = encode(batch)
        
        if trans is not None:
            trans_1 = trans.clone()
        if rotmats is not None:
            rotmats_1 = rotmats.clone()

        pos_ha, _, _ = full_atom_reconstruction(R_bb=rotmats_1, t_bb=trans_1, angles=angles_1, aa=seqs_1)
        pos_ha = F.pad(pos_ha, pad=(0, 0, 0, 15-14), value=0.).to(self.device)
        
        pos_heavyatom_clone = batch['pos_heavyatom'].clone()
        pos_new = torch.where(batch['generate_mask'][:, :, None, None].to(self.device), pos_ha.to(self.device), pos_heavyatom_clone.to(self.device))
        
        coords, atnames = convert_to_custom_format({
            'aa': batch['aa'],
            'chain_nb': batch['chain_nb'],
            'pos_heavyatom': pos_new,
        })

        assert not torch.isnan(coords).any(), "coords contains NaN values"

        info_tensors = dataStructures.create_info_tensors(atnames, device=self.device)

        energy = self.forceField_Obj(coords.to(self.device), info_tensors)

        return torch.mean(torch.sum(energy[:, :, :, 0], dim=-1))
    
def remove_module_prefix(state_dict):
    return {k.replace('module.', ''): v for k, v in state_dict.items()}

def oc_so3_opt(model, R, batch, trans_0, rotmats_0, device, u_positions, alpha=0.7, beta=1.5, max_iter=20, n_sample=0):
    model.to(device)
    t_N = u_positions[-1]
    u_N = len(u_positions) - 1
    with torch.no_grad():
        rotmats_gt, trans_gt, angles_gt, seqs_gt, node_embed, edge_embed = model.encode(batch)
        gen_mask, res_mask = batch['generate_mask'].long(), batch['res_mask'].long()
    E_s = [torch.tensor([[0.,0.,0.],[0.,0.,-1.],[0.,1.,0.]]).to(device), torch.tensor([[0.,0.,1.],[0.,0.,0.],[-1.,0.,0.]]).to(device), torch.tensor([[0.,-1.,0.],[1.,0.,0.],[0.,0.,0.]]).to(device)]
    ts = torch.linspace(1.e-2, 1.0, t_N+1).to(device) 
    num_batch = batch['aa'].shape[0]
    
    def dynamic_rot(x, t, r_trans, dt):
        pred_rotmats_1, _, _, _ = model.ga_encoder(t, x, r_trans, angles_gt, seqs_gt, node_embed, edge_embed, batch['generate_mask'].long(), batch['res_mask'].long())
        rot_vf = so3_utils.calc_rot_vf(x, pred_rotmats_1)
        return so3_utils.vector_to_skew_matrix(rot_vf)

    def dynamic_trans(t, x, r_rotmats):
        _, pred_trans_1, _, _ = model.ga_encoder(t,r_rotmats, x, angles_gt, seqs_gt, node_embed, edge_embed, batch['generate_mask'].long(), batch['res_mask'].long())
        return pred_trans_1-trans_0
    
    @torch.no_grad()
    def generate_traj(u_trans, u_rotmats):
        r_trans = trans_0.detach().clone()
        r_trans = torch.where(batch['generate_mask'][..., None], r_trans, trans_gt)
        r_rotmats = rotmats_0.detach().clone()
        r_rotmats = torch.where(batch['generate_mask'][...,None,None],r_rotmats, rotmats_gt)
        outs_trans, outs_rotmat = [r_trans], [r_rotmats]

        t_1 = ts[0]
        for i, t_2 in enumerate(ts[1:]):
            t = torch.ones((num_batch, 1), device=device)*t_1
            d_t = (t_2 - t_1) * torch.ones((num_batch, 1), device=device)
            pred_rotmats_1, pred_trans_1, _, _ = model.ga_encoder(t, r_rotmats, r_trans, angles_gt, seqs_gt, node_embed, edge_embed, batch['generate_mask'].long(), batch['res_mask'].long())
            pred_rotmats_1 = torch.where(batch['generate_mask'][...,None,None],pred_rotmats_1, rotmats_gt)
            pred_trans_1 = torch.where(batch['generate_mask'][...,None],pred_trans_1,trans_gt)
            if i in u_positions:
                r_trans = r_trans + (pred_trans_1 - trans_0 + u_trans[u_positions.index(i)]) * d_t[...,None]
            else:
                r_trans = r_trans + (pred_trans_1 - trans_0) * d_t[...,None]
            r_trans = torch.where(batch['generate_mask'][...,None],r_trans,trans_gt)
            rot_vf = so3_utils.calc_rot_vf(r_rotmats, pred_rotmats_1)
            if i in u_positions:
                sum_vf = rot_vf+so3_utils.rotmat_to_rotvec(u_rotmats[u_positions.index(i)])
                r_rotmats = r_rotmats @ so3_utils.rotvec_to_rotmat(d_t*10 * sum_vf)
            else:
                mat_t = so3_utils.rotvec_to_rotmat(d_t*10*rot_vf)
                r_rotmats = r_rotmats @ mat_t
            r_rotmats = torch.where(batch['generate_mask'][...,None,None], r_rotmats, rotmats_gt)
            outs_trans.append(r_trans)
            outs_rotmat.append(r_rotmats)
            t_1 = t_2

        t_1 = ts[-1]
        t = torch.ones((num_batch, 1), device=device) * t_1
        pred_rotmats_1, pred_trans_1,_, _ = model.ga_encoder(t, r_rotmats, r_trans, angles_gt, seqs_gt, node_embed, edge_embed, batch['generate_mask'].long(), batch['res_mask'].long())
        pred_rotmats_1 = torch.where(batch['generate_mask'][...,None,None],pred_rotmats_1,rotmats_gt)
        pred_trans_1 = torch.where(batch['generate_mask'][...,None],pred_trans_1,trans_gt)
        return outs_trans, outs_rotmat, pred_trans_1, pred_rotmats_1
    
    @torch.no_grad()
    def generate_traj_mu(mu_rotmat_T, u_rotmats, trans_traj, rotmats_traj):
        mu_rotmats = mu_rotmat_T
        
        def solve_mu(A, B_list, E_list, c):
            """
            Solves for u in the equation: u = A - c * sum(u_i * E_i)
    
            Arguments:
            A -- Tensor of shape (batch_size, num_matrices, 3, 3), constant matrix A
            B_list -- List of 3 tensors, each of shape (batch_size, num_matrices, 3, 3), constant matrices B_1, B_2, B_3
            E_list -- List of 3 tensors, each of shape (3, 3), basis matrices E_1, E_2, E_3
            c -- Constant scalar
            Returns:
            u -- Solution tensor of shape (batch_size, num_matrices, 3, 3)
            """
            batch_size, num_matrices, _, _ = A.shape
            B_vec = torch.stack([B.reshape(batch_size, num_matrices, 9) for B in B_list], dim=-1)
            E_vec = torch.stack([E.reshape(9) for E in E_list], dim=-1)
            E_vec = E_vec.unsqueeze(0).unsqueeze(0)
            I = torch.eye(9).to(A.device).unsqueeze(0).unsqueeze(0) 
            cEBT = c * E_vec @ B_vec.transpose(-2,-1)
            transformation_matrix = I + cEBT
            transformation_matrix_inv = torch.inverse(transformation_matrix) 
            A_vec = A.reshape(batch_size, num_matrices, 9) 
            u_vec = transformation_matrix_inv@A_vec.unsqueeze(-1)
            u_vec = u_vec.squeeze(-1) 
            u = u_vec.reshape(batch_size, num_matrices, 3, 3) 
            return u

        mu_rotmats_s = []
        for i in range(u_N):
            d_t = (ts[u_positions[u_N-i]]-ts[u_positions[u_N-1-i]]) * torch.ones((num_batch, 1), device=device)
            t = ts[u_positions[u_N-1-i]]* torch.ones((num_batch, 1), device=device)
            rotmat = rotmats_traj[u_positions[u_N-1-i]]
            trans = trans_traj[u_positions[u_N-1-i]]
            rotmat.requires_grad = (True)
            multi_s = []
            for j in range(3):
                func = lambda x: dynamic_rot(x, t, trans, d_t)
                Df_x_Ei = jvp(func, (rotmat,), (rotmat@E_s[j],))[1]
                vf_rotmats = func(rotmat)
                u_rotmat = u_rotmats[u_N-1-i]
                multi_s.append(-((vf_rotmats+u_rotmat)@E_s[j]-E_s[j]@(vf_rotmats+u_rotmat) + Df_x_Ei))
            mu_rotmats = solve_mu(mu_rotmats, multi_s, E_s, d_t/2)
            mu_rotmats_s.append(mu_rotmats)
        return mu_rotmats_s

    u_ind = [i for i in range(u_N)]
    u_trans, u_rotmats = {}, {}

    for ind in u_ind:
        u_rotmats[ind] = torch.zeros_like(rotmats_0).to(device)
        u_rotmats[ind].requires_grad = True
        u_rotmats[ind].grad = torch.zeros_like(u_rotmats[ind], device=device)
        
        u_trans[ind] = torch.zeros_like(trans_0).to(device)
        u_trans[ind].requires_grad = True
        u_trans[ind].grad = torch.zeros_like(u_trans[ind], device=device)

    pred_rotmats_1, pred_trans_1, trans_1, rotmats_1 = None, None, None, None
    for n in range(max_iter):
        trans_traj, rotmats_traj, _, _ = generate_traj(u_trans, u_rotmats)
        trans_1, rotmats_1 = trans_traj[-1], rotmats_traj[-1]
        trans_1.requires_grad_(True)
        rotmats_1.requires_grad_(True)
        t = torch.ones((num_batch, 1), device=device) * ts[-1]
        pred_rotmats_1, pred_trans_1, _, _ = model.ga_encoder(t, rotmats_1, trans_1, angles_gt, seqs_gt, node_embed, edge_embed, gen_mask, res_mask)
        pred_rotmats_1 = torch.where(batch['generate_mask'][...,None,None],pred_rotmats_1,rotmats_gt)
        pred_trans_1 = torch.where(batch['generate_mask'][...,None],pred_trans_1,trans_gt)
        loss = - R.get_value(pred_trans_1, pred_rotmats_1, batch)
        reg_trans = (pred_trans_1-trans_gt).abs().mean()
        def rotation_angle(R_pred, R_gt):
            R_diff = torch.matmul(R_pred, R_gt.transpose(-1, -2))
            trace = R_diff.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)
            cos_theta = (trace - 1) / 2
            cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
            theta = torch.acos(cos_theta)
            return theta
        theta = rotation_angle(pred_rotmats_1, rotmats_gt)
        reg_rot = theta.mean()
        loss = loss - reg_trans - reg_rot * 0
        print(f"{batch['id'][0]} iter {n} is {-loss};reg_trans:{reg_trans};reg_rot:{reg_rot}", flush=True)

        des_trans_grad = torch.autograd.grad(loss, trans_1, retain_graph=True)[0].to(device)
        des_rotmats_grad = torch.autograd.grad(loss, rotmats_1, retain_graph=True)[0].to(device)
                        
        mu_rotmat_T = torch.zeros_like(rotmats_0).to(device)
        for i in range(3):
            mu_rotmat_T += torch.einsum('...ii->...', des_rotmats_grad.transpose(-2,-1) @ (rotmats_1@E_s[i]))[...,None,None] / 2 * E_s[i]

        mu_rotmats_s = generate_traj_mu(mu_rotmat_T, u_rotmats, trans_traj, rotmats_traj)

        lam = des_trans_grad
        for j in range(u_N - 1, 0, -1):
            z_j = trans_traj[j]
            d_t = (ts[u_positions[j+1]]-ts[u_positions[j]]) * torch.ones((num_batch, 1), device=device)
            t = ts[u_positions[j]]* torch.ones((num_batch, 1), device=device)

            func = lambda x: (x.contiguous().view_as(z_j) + 
                            u_trans[j-1].detach().clone().to(device) + 
                            dynamic_trans(t, x.contiguous().view_as(z_j) + 
                            u_trans[j-1].detach().clone().to(device), rotmats_traj[j]) * d_t).view(-1)

            _, vjp = torch.autograd.functional.vjp(func, z_j.view(-1), v=lam.view(-1).to(device))
            lam = vjp.detach().clone().view_as(z_j)
            lam = torch.where(torch.isnan(lam) | torch.isinf(lam), torch.mean(lam), lam)
            u_trans[j-1].grad = torch.where(batch['generate_mask'][...,None], lam.detach().clone() ,torch.zeros_like(u_trans[j-1]))
        for i in range(u_N-1):
            u_rotmats[i] = u_rotmats[i] * alpha + beta * mu_rotmats_s[u_N - 1 - i]
            u_rotmats[i] = torch.where(batch['generate_mask'][...,None,None], u_rotmats[i], torch.zeros_like(u_rotmats[i]))
            u_trans[i] = alpha * u_trans[i] + beta * u_trans[i].grad
            u_trans[i] = torch.where(batch['generate_mask'][...,None], u_trans[i], torch.zeros_like(u_trans[i]))
    
    with torch.no_grad():
        _, _, trans_T, rotmats_T = generate_traj(u_trans, u_rotmats)
        energy = R.get_value(trans_T, rotmats_T, batch)
        print(f"{batch['id'][0]}: Energy after optimization is {energy}", flush=True)
    return trans_T, rotmats_T, energy
  
def oc_so3_as_eu(model, R, batch, trans_0, rotmats_0, device, u_positions, alpha=0.7, beta=1.5, max_iter=20, n_sample=0, arg_reg_rot=0):
    model.to(device)
    t_N = u_positions[-1]
    u_N = len(u_positions) - 1
    with torch.no_grad():
        rotmats_gt, trans_gt, angles_gt, seqs_gt, node_embed, edge_embed = model.encode(batch)
        gen_mask, res_mask = batch['generate_mask'].long(), batch['res_mask'].long()
    ts = torch.linspace(1.e-2, 1.0, t_N+1).to(device) # change dt
    num_batch = batch['aa'].shape[0]

    def dynamic_rot(t, r_trans, x):
        pred_rotmats_1, _, _, _ = model.ga_encoder(t, x, r_trans, angles_gt, seqs_gt, node_embed, edge_embed, batch['generate_mask'].long(), batch['res_mask'].long())
        rot_vf = so3_utils.calc_rot_vf(x, pred_rotmats_1)
        return so3_utils.vector_to_skew_matrix(rot_vf)

    def generate_traj(u_trans, u_rotmats):
        r_trans = trans_0.detach().clone()
        r_trans = torch.where(batch['generate_mask'][..., None], r_trans, trans_gt)
        r_rotmats = rotmats_0.detach().clone()
        r_rotmats = torch.where(batch['generate_mask'][..., None, None], r_rotmats, rotmats_gt)
        outs_trans, outs_rotmat = [], []

        t_1 = ts[0]
        for i, t_2 in enumerate(ts[1:]):
            t = torch.ones((num_batch, 1), device=device) * t_1
            d_t = (t_2 - t_1) * torch.ones((num_batch, 1), device=device)
            with torch.no_grad():
                pred_rotmats_1, pred_trans_1, _, _ = model.ga_encoder(
                    t, r_rotmats, r_trans, angles_gt, seqs_gt,
                    node_embed, edge_embed, 
                    batch['generate_mask'].long(), batch['res_mask'].long()
                )
                pred_rotmats_1 = torch.where(batch['generate_mask'][..., None, None], pred_rotmats_1, rotmats_gt)
                pred_trans_1 = torch.where(batch['generate_mask'][..., None], pred_trans_1, trans_gt)

            rot_vf = so3_utils.calc_rot_vf(r_rotmats, pred_rotmats_1.detach())
            if i in u_positions:
                sum_vf = rot_vf + so3_utils.skew_matrix_to_vector(u_rotmats[u_positions.index(i)])
                r_rotmats = r_rotmats @ so3_utils.rotvec_to_rotmat(d_t * 10 * sum_vf)
            else:
                mat_t = so3_utils.rotvec_to_rotmat(d_t * 10 * rot_vf)
                r_rotmats = r_rotmats @ mat_t

            r_rotmats = torch.where(batch['generate_mask'][..., None, None], r_rotmats, rotmats_gt)
            with torch.no_grad():
                outs_trans.append(r_trans.cpu())
                outs_rotmat.append(r_rotmats.cpu())
            
            t_1 = t_2
            del pred_rotmats_1, pred_trans_1 
            torch.cuda.empty_cache()

        t_1 = ts[-1]
        t = torch.ones((num_batch, 1), device=device) * t_1
        pred_rotmats_1, pred_trans_1, _, _ = model.ga_encoder(
            t, r_rotmats, r_trans, angles_gt, seqs_gt,
            node_embed, edge_embed, 
            batch['generate_mask'].long(), batch['res_mask'].long()
        )
        pred_rotmats_1 = torch.where(batch['generate_mask'][..., None, None], pred_rotmats_1, rotmats_gt)
        pred_trans_1 = torch.where(batch['generate_mask'][..., None], pred_trans_1, trans_gt)

        return outs_trans, outs_rotmat, pred_trans_1, pred_rotmats_1

    u_ind = [i for i in range(u_N)]
    u_trans, u_rotmats = {}, {}

    for ind in u_ind:
        u_rotmats[ind] = torch.zeros_like(rotmats_0).to(device)
        u_rotmats[ind].requires_grad = True
        u_rotmats[ind].grad = torch.zeros_like(u_rotmats[ind], device=device)

    pred_rotmats_1, pred_trans_1, trans_1, rotmats_1 = None, None, None, None
    for n in range(max_iter):
        trans_traj, rotmats_traj, pred_trans_1, pred_rotmats_1 = generate_traj(u_trans, u_rotmats)
        loss = - R.get_value(pred_trans_1, pred_rotmats_1, batch)
        for ind in u_ind:
            if u_rotmats[ind].requires_grad:
                grad_rotmat = torch.autograd.grad(loss, u_rotmats[ind], retain_graph=True, allow_unused=True)[0]
                grad_rotmat = torch.where(batch['generate_mask'][...,None,None], grad_rotmat, torch.zeros_like(grad_rotmat))
                grad_rotmat = grad_rotmat.detach().clone()
                grad_rotmat = torch.where(torch.isnan(grad_rotmat) | torch.isinf(grad_rotmat), torch.mean(grad_rotmat), grad_rotmat)
                if grad_rotmat is not None:
                    u_rotmats[ind].grad = grad_rotmat
                torch.cuda.empty_cache()
        for i in range(u_N-1):
            if torch.isnan(u_rotmats[i].grad).any():
                print(f"{n}-{i} is gradnan: {u_rotmats[i].grad}")
            u_rotmats[i] = u_rotmats[i] * alpha + beta * u_rotmats[i].grad
            u_rotmats[i] = so3_utils.vector_to_skew_matrix(so3_utils.skew_matrix_to_vector(u_rotmats[i]))
            u_rotmats[i] = torch.where(batch['generate_mask'][...,None,None], u_rotmats[i], torch.zeros_like(u_rotmats[i]))
    
    with torch.no_grad():
        _, _, trans_T, rotmats_T = generate_traj(u_trans, u_rotmats)
        energy = R.get_value(trans_T, rotmats_T, batch)
        print(f"{batch['id'][0]}: Energy after optimization is {energy}", flush=True)
    return trans_T, rotmats_T, energy

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='rlhf_fm/peptide/configs/learn_angle.yaml')
    parser.add_argument('--logdir', type=str, default="rlhf_fm/peptide/rlhf_finetune/sample10_trans_reg")
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--reg_rot', type=int, default=0)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--tag', type=str, default='')
    # rlhf
    parser.add_argument('--resume', type=str, default="dataset/model1.pt")
    parser.add_argument('--algorithm', type=str, default='oc_opt')
    parser.add_argument('--name', type=str, default='pepflow')
    parser.add_argument('--alpha',type=float, default=0.7)
    parser.add_argument('--beta', type=float, default=0.2)
    parser.add_argument('--start_data', type=int, default=0)
    parser.add_argument('--n_tasks', type=int, default=10)
    parser.add_argument('--todo_data', type=int, default=0)
    args = parser.parse_args()
    args.logdir = f"{args.logdir}_{args.alpha}_{args.beta}_regrot_{args.reg_rot}"

    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device is {device}")
    # Load configs
    config, config_name = load_config(args.config)
    seed_all(config.train.seed)
    config['device'] = device
    
    # Logging
    if args.debug:
        logger = get_logger('train', None)
        writer = BlackHole()
    else:
        run = wandb.init(project=args.name, config=config, name='%s[%s]%s' % (config_name, args.tag, timestamp))
        log_dir = os.path.dirname(os.path.dirname(args.resume))
        ckpt_dir = os.path.join(log_dir, 'checkpoints')
        if not os.path.exists(ckpt_dir): os.makedirs(ckpt_dir)
        logger = get_logger('train', log_dir)
        if not os.path.exists(os.path.join(log_dir, os.path.basename(args.config))):
            shutil.copyfile(args.config, os.path.join(log_dir, os.path.basename(args.config)))
    print(args)
    print(config)

    # Data
    print('Loading datasets...')
    dataset = PepDataset(structure_dir = config.dataset.val.structure_dir, dataset_dir = config.dataset.val.dataset_dir,
                                            name = config.dataset.val.name, transform=None, reset=config.dataset.val.reset)
    print(f"len of dataset is {len(dataset)}")
    def get_rank_data_indices(rank, num_ranks, dataset_len):
        chunk_size = dataset_len // num_ranks
        remainder = dataset_len % num_ranks
        start = rank * chunk_size + min(rank, remainder)
        end = start + chunk_size + (1 if rank < remainder else 0)
        return start, end
    
    if args.todo_data == 0:
        args.todo_data = len(dataset)
    start, end = get_rank_data_indices(rank, args.n_tasks, args.todo_data)
    rank_dataset = Subset(dataset, range(start, end))
    dataloader = DataLoader(rank_dataset, batch_size=1, shuffle=False, collate_fn=PaddingCollate(eight=False), num_workers=4, pin_memory=True)
    print(f'Test set {len(rank_dataset)}')

    # Model
    print('Building model...')
    model = FlowModel(config.model).to(device)
    print(f'Number of parameters: {count_parameters(model)}')


    # Resume
    if args.resume is not None:
        print('Resuming from checkpoint: %s' % args.resume)
        ckpt = torch.load(args.resume, map_location=device)
        # print("Keys in checkpoint:", ckpt['model'].keys())
        it_first = ckpt['iteration']  # + 1
        model.load_state_dict(process_dic(ckpt['model']))
        # print('Resuming optimizer states...')
        # optimizer.load_state_dict(ckpt['optimizer'])
        # print('Resuming scheduler states...')
        # scheduler.load_state_dict(ckpt['scheduler'])

    R = Reward()

    dic = {'id':[],'n':[], 'len':[],'tran':[],'rot':[],'ft_tran':[], 'ft_rot':[], 'n_sample':[], 'gt_madrax':[], 'sp_madrax':[], 'ft_madrax':[],}
    # Main training loop
    for i, batch in enumerate(tqdm(dataloader, desc='finetuning...', dynamic_ncols=True)):
        print(batch['id'][0])
        avg_time = 0
        for n in range(10):
            file_path = os.path.join(args.logdir, f"pdb/{batch['id'][0]}/opt_{n}.pdb")
            if os.path.exists(file_path):
                continue
            batch = recursive_to(batch, device)
            with torch.no_grad():
                traj_1, trans_0, rotmats_0= model.sample(batch,num_steps=200,sample_bb=True,sample_ang=False,sample_seq=False)
                ca_dist = torch.sqrt(torch.sum((traj_1[-1]['trans']-traj_1[-1]['trans_1'])**2*batch['generate_mask'][...,None].cpu().long()) / (torch.sum(batch['generate_mask']) + 1e-8).cpu())
                rot_dist = torch.sqrt(torch.sum((traj_1[-1]['rotmats']-traj_1[-1]['rotmats_1'])**2*batch['generate_mask'][...,None,None].long().cpu()) / (torch.sum(batch['generate_mask']) + 1e-8).cpu())
                print(f"{batch['id'][0]}:tran:{ca_dist},rot:{rot_dist},len:{batch['generate_mask'].sum().item()}")
                samples = traj_1[-1]
                samples['batch'] = batch
                pos_ha,_,_ = full_atom_reconstruction(R_bb=samples['rotmats'].to('cpu'),t_bb=samples['trans'].to('cpu'),angles=samples['angles'].to('cpu'),aa=samples['seqs'].to('cpu'))
                pos_ha = F.pad(pos_ha, pad=(0,0,0,15-14), value=0.).to(device)
                pos_new = torch.where(batch['generate_mask'][:,:,None,None],pos_ha,batch['pos_heavyatom'])
                rmsd = torch.sqrt(((pos_new[:, :, :4, :] - batch['pos_heavyatom'][:, :, :4, :]) ** 2).mean(dim=-1))
                ori_rmsd = torch.mean(rmsd[batch['generate_mask']])
                gt_score = R.get_value(trans=None, rotmats=None, batch=batch)
                sp_score = R.get_value(trans=samples['trans'].to(device), rotmats=samples['rotmats'].to(device), batch=batch)
                print(f"{batch['id'][0]}: gt {gt_score} vs sample {sp_score}", flush=True)

            # Run optimization
            # sample the initial state
            if args.algorithm == "oc_so3_opt":
                u_positions = [0,20,40,60,80,100,120,140,160,180,199]
                trans_1, rotmats_1, final_score = oc_so3_opt(
                    model, R, batch=batch, trans_0=trans_0,rotmats_0=rotmats_0, device=device, u_positions=u_positions, alpha=args.alpha, beta=args.beta, max_iter=10,n_sample=n, arg_reg_rot=args.reg_rot
                )
            # save the optimized sample and calculate metrics
            ft_ca_dist = torch.sqrt(torch.sum((trans_1.cpu()-traj_1[-1]['trans_1'])**2*batch['generate_mask'][...,None].cpu().long()) / (torch.sum(batch['generate_mask']) + 1e-8).cpu())
            ft_rot_dist = torch.sqrt(torch.sum((rotmats_1.cpu()-traj_1[-1]['rotmats_1'])**2*batch['generate_mask'][...,None,None].long().cpu()) / (torch.sum(batch['generate_mask']) + 1e-8).cpu())
            pos_ha,_,_ = full_atom_reconstruction(R_bb=rotmats_1.to(device),t_bb=trans_1.to(device),angles=samples['angles'].to(device),aa=samples['seqs'].to(device))
            pos_ha = F.pad(pos_ha, pad=(0,0,0,15-14), value=0.).to(device)
            pos_new = torch.where(batch['generate_mask'][:,:,None,None],pos_ha,batch['pos_heavyatom'])
            batch_opt = batch.copy()
            batch_opt['pos_heavyatom'] = pos_new
            os.makedirs(os.path.join(args.logdir, f"samples"),exist_ok=True)
            os.makedirs(os.path.join(args.logdir, f"pdb/{batch['id'][0]}"),exist_ok=True)
            torch.save(batch_opt,f'{args.logdir}/samples/{batch["id"][0]}_opt{n}.pt')
            save_samples_sc_opt(batch, os.path.join(args.logdir, f"pdb/{batch['id'][0]}"), pos_new,n)
            torch.save(batch,f'{args.logdir}/samples/{batch["id"][0]}_gt.pt')
            torch.save(traj_1[-1],f'{args.logdir}/samples/{batch["id"][0]}_sample_{n}.pt')
            save_samples_sc(samples, os.path.join(args.logdir, f"pdb/{batch['id'][0]}"),n)
            dic['n'].append(n)
            dic['tran'].append(ca_dist.item())
            dic['rot'].append(rot_dist.item())
            dic['ft_tran'].append(ft_ca_dist.item())
            dic['ft_rot'].append(ft_rot_dist.item())
            dic['id'].append(batch['id'][0])
            dic['len'].append(batch['generate_mask'].sum().item())
            dic['n_sample'].append(n)
            dic['gt_madrax'].append(gt_score.item())
            dic['sp_madrax'].append(sp_score.item())
            dic['ft_madrax'].append(final_score.item())
            torch.cuda.empty_cache()
            gc.collect()
    output_file = os.path.join(args.logdir, 'outputs.csv')
    df = pd.DataFrame(dic)

    if not os.path.exists(output_file):
        df.to_csv(output_file, mode='w', index=False, header=True)
    else:
        df.to_csv(output_file, mode='a', index=False, header=False)