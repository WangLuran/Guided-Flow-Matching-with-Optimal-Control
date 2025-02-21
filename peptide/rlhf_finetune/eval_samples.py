# C_alpha RMSD, SSR, BSR
# pyrosetta: stability, affinity
# Designability, Diversity
from eval.geometry import get_bind_ratio, get_rmsd, get_chain_from_pdb, get_traj_chain, get_ss, get_tm
from eval.energy import get_rosetta_score_base
import os, sys
import time
import json
from madrax.ForceField import ForceField
from madrax import utils,dataStructures
import time,os, shutil
import torch
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
os.environ["CUDA_VISIBLE_DEVICES"] = str(rank % 2)
print(f"Process {rank} is using GPU {os.environ['CUDA_VISIBLE_DEVICES']}")
import argparse
parser = argparse.ArgumentParser(description='eval')
parser.add_argument('--n_tasks', type=int, default=5)
parser.add_argument('--log_dir', type=str, default='')
parser.add_argument('--start_data', type=int, default=0)
parser.add_argument('--num_folders', type=int, default=0)
args = parser.parse_args()
size = args.n_tasks
log_dir = args.log_dir
folders = [folder for folder in os.listdir(log_dir) if os.path.isdir(os.path.join(log_dir, folder))]

max_folders = len(folders)
if args.num_folders == 0:
    num_folders = len(folders)
else:
    num_folders = args.num_folders
folders_per_rank = num_folders // size
remaining_folders = num_folders % size

start_index = rank * folders_per_rank + min(rank, remaining_folders) + args.start_data
end_index = min(max_folders,start_index + folders_per_rank + (1 if rank < remaining_folders else 0))

print(f"evaluating from {start_index} to {end_index}", flush=True)
assigned_folders = folders[start_index:end_index]
print(assigned_folders, flush=True)
total_folders = len(assigned_folders)

def get_madrax(pdb_path, device="cpu"):
    parent_dir, pdb_file = os.path.split(pdb_path)
    dir_to_create = os.path.splitext(pdb_file)[0]
    new_dir_path = os.path.join(parent_dir, dir_to_create)
    try:
        if not os.path.exists(new_dir_path):
            os.makedirs(new_dir_path)
        
        new_pdb_path = os.path.join(new_dir_path, pdb_file)
        try:
            shutil.copy(pdb_path, new_pdb_path)
        except:
            print("file already exists", flush=True)
        coords, atnames, pdbNames = utils.parsePDB(new_dir_path)
        info_tensors = dataStructures.create_info_tensors(atnames, device=device)
        forceField_Obj = ForceField(device=device)
        energy = forceField_Obj(coords.to(device), info_tensors)
        
        result = torch.mean(torch.sum(energy[:, :, :, 0], dim=-1))
        return result.item()
    
    finally:
        if os.path.exists(new_dir_path):
            shutil.rmtree(new_dir_path)

for index, folder in enumerate(assigned_folders):
    start_time = time.time()
    print(f'Processing folder {index + 1}/{total_folders} in rank-{rank}: {folder}', file=sys.stderr, flush=True)
    if os.path.isdir(os.path.join(log_dir, folder)):
        output_path = os.path.join(log_dir, folder, 'result.txt')
        output_all_path = os.path.join(log_dir, folder, 'result_mean.txt')
        sp_div, opt_div = 0, 0
        chain_id = folder.split("_")[-1]
        if os.path.exists(output_path):
            print(f"loading precalculated {output_path}",flush=True)
            with open(output_path, 'r') as file:
                dic = json.load(file)
        else:
            print(f"calculating {output_path}",flush=True)
            gt_path = os.path.join(log_dir, folder, "gt.pdb")
            gt_madrax = get_madrax(gt_path)
            if not os.path.exists(gt_path):
                continue
            gt_chain = get_chain_from_pdb(gt_path, chain_id)
            gt_rosetta =  get_rosetta_score_base(gt_path, chain_id)
            dic = {'id':[], 'n':[], 'sp_imp':[], 'opt_imp':[], 'gt_madrax':[], 'sp_madrax':[], 'opt_madrax':[], 'sp_rmsd1':[], 'sp_rmsd2':[], 'opt_rmsd1':[],'opt_rmsd2':[],'sp_ssr':[],'opt_ssr':[], 'bsr_sp':[], 'bsr_opt':[],'gt_stab':[], 'sp_stab':[], 'opt_stab':[], 'gt_bind':[], 'sp_bind':[], 'opt_bind':[]}
            for i in range(32):
                sp_path = os.path.join(log_dir, folder, f"sample_{i}.pdb")
                opt_path = os.path.join(log_dir, folder, f"opt_{i}.pdb")
                if not os.path.exists(sp_path) or not os.path.exists(opt_path):
                    continue
                dic['id'].append(folder)
                dic['n'].append(str(i))
                sp_chain = get_chain_from_pdb(sp_path, chain_id)
                opt_chain = get_chain_from_pdb(opt_path, chain_id)

                dic['gt_madrax'].append(gt_madrax)
                dic['sp_madrax'].append(get_madrax(sp_path))
                dic['opt_madrax'].append(get_madrax(opt_path))

                sp_rmsd = get_rmsd(gt_chain, sp_chain)
                opt_rmsd = get_rmsd(gt_chain, opt_chain)
                dic['sp_rmsd1'].append(sp_rmsd[0])
                dic['sp_rmsd2'].append(sp_rmsd[1])
                dic['opt_rmsd1'].append(opt_rmsd[0])
                dic['opt_rmsd2'].append(opt_rmsd[1])
                
                sp_ssr = get_ss(get_traj_chain(gt_path, chain_id), get_traj_chain(sp_path, chain_id))
                opt_ssr = get_ss(get_traj_chain(gt_path, chain_id), get_traj_chain(opt_path, chain_id))
                dic['sp_ssr'].append(sp_ssr)
                dic['opt_ssr'].append(opt_ssr)
                
                bsr_sp = get_bind_ratio(gt_path, sp_path, chain_id, chain_id)
                bsr_opt = get_bind_ratio(gt_path, opt_path, chain_id, chain_id)
                dic['bsr_sp'].append(bsr_sp)
                dic['bsr_opt'].append(bsr_opt)

                print("begin to calculate energy", flush=True)
                sp_rosetta = get_rosetta_score_base(sp_path, chain_id)
                opt_rosetta = get_rosetta_score_base(opt_path, chain_id)
                dic['gt_stab'].append(gt_rosetta['stab'])
                dic['sp_stab'].append(sp_rosetta['stab'])
                dic['opt_stab'].append(opt_rosetta['stab'])
                dic['gt_bind'].append(gt_rosetta['bind'])
                dic['sp_bind'].append(sp_rosetta['bind'])
                dic['opt_bind'].append(opt_rosetta['bind'])
                if sp_rosetta['bind']<=gt_rosetta['bind']:
                    dic['sp_imp'].append(1)
                else:
                    dic['sp_imp'].append(0)
                if opt_rosetta['bind']<=gt_rosetta['bind']:
                    dic['opt_imp'].append(1)
                else:
                    dic['opt_imp'].append(0)

        cnt = 0
        for i in range(10):
            for j in range(i+1, 10):
                sp_i_path = os.path.join(log_dir, folder, f"sample_{i}.pdb")
                opt_i_path = os.path.join(log_dir, folder, f"opt_{i}.pdb")
                sp_j_path = os.path.join(log_dir, folder, f"sample_{j}.pdb")
                opt_j_path = os.path.join(log_dir, folder, f"opt_{j}.pdb")
                if not os.path.exists(sp_i_path) or not os.path.exists(sp_j_path):
                    continue
                sp_i_chain = get_chain_from_pdb(sp_i_path, chain_id)
                opt_i_chain = get_chain_from_pdb(opt_i_path, chain_id)
                sp_j_chain = get_chain_from_pdb(sp_j_path, chain_id)
                opt_j_chain = get_chain_from_pdb(opt_j_path, chain_id)
                sp_div += get_tm(sp_i_chain, sp_j_chain)
                opt_div += get_tm(opt_i_chain, opt_j_chain)
                cnt += 1
        sp_div = sp_div/cnt
        opt_div = opt_div/cnt

        mean_dict = {
            'sp_div': 1-sp_div,
            'opt_div': 1-opt_div,
            'sp_imp': sum(dic['sp_imp']) / len(dic['sp_imp']) if dic['sp_imp'] else 0,
            'opt_imp': sum(dic['opt_imp']) / len(dic['opt_imp']) if dic['opt_imp'] else 0,
            'gt_madrax_mean': sum(dic['gt_madrax']) / len(dic['gt_madrax']) if dic['gt_madrax'] else 0,
            'sp_madrax_mean': sum(dic['sp_madrax']) / len(dic['sp_madrax']) if dic['sp_madrax'] else 0,
            'opt_madrax_mean': sum(dic['opt_madrax']) / len(dic['opt_madrax']) if dic['opt_madrax'] else 0,
            'sp_rmsd1_mean': sum(dic['sp_rmsd1']) / len(dic['sp_rmsd1']) if dic['sp_rmsd1'] else 0,
            'sp_rmsd2_mean': sum(dic['sp_rmsd2']) / len(dic['sp_rmsd2']) if dic['sp_rmsd2'] else 0,
            'opt_rmsd1_mean': sum(dic['opt_rmsd1']) / len(dic['opt_rmsd1']) if dic['opt_rmsd1'] else 0,
            'opt_rmsd2_mean': sum(dic['opt_rmsd2']) / len(dic['opt_rmsd2']) if dic['opt_rmsd2'] else 0,
            'sp_ssr_mean': sum(dic['sp_ssr']) / len(dic['sp_ssr']) if dic['sp_ssr'] else 0,
            'opt_ssr_mean': sum(dic['opt_ssr']) / len(dic['opt_ssr']) if dic['opt_ssr'] else 0,
            'bsr_sp_mean': sum(dic['bsr_sp']) / len(dic['bsr_sp']) if dic['bsr_sp'] else 0,
            'bsr_opt_mean': sum(dic['bsr_opt']) / len(dic['bsr_opt']) if dic['bsr_opt'] else 0,
            'gt_stab_mean': sum(dic['gt_stab']) / len(dic['gt_stab']) if dic['gt_stab'] else 0,
            'sp_stab_mean': sum(dic['sp_stab']) / len(dic['sp_stab']) if dic['sp_stab'] else 0,
            'opt_stab_mean': sum(dic['opt_stab']) / len(dic['opt_stab']) if dic['opt_stab'] else 0,
            'gt_bind_mean': sum(dic['gt_bind']) / len(dic['gt_bind']) if dic['gt_bind'] else 0,
            'sp_bind_mean': sum(dic['sp_bind']) / len(dic['sp_bind']) if dic['sp_bind'] else 0,
            'opt_bind_mean': sum(dic['opt_bind']) / len(dic['opt_bind']) if dic['opt_bind'] else 0
        }
        print("saving mean_values", flush=True)
        with open(output_path, 'w') as file:
            json.dump(dic, file)
        with open(output_all_path, 'w') as file:
            json.dump(mean_dict, file)
        end_time = time.time()
        print(f"calculating time is {end_time - start_time}")