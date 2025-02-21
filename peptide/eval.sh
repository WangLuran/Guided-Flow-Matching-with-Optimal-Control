#!/bin/bash
#SBATCH --job-name=eval_so3

cd rlhf_fm/peptide

source ~/.bashrc
conda activate flow

export PYTHONPATH=$(pwd):$PYTHONPATH

mpirun -np $SLURM_NTASKS python rlhf_finetune/eval_samples.py --n_tasks 1 --start_data 0  --num_folders 30 --log_dir "rlhf_fm/peptide/exp_rebuttal/so3_in_eu_0.95_2.0_regrot_0/pdb"