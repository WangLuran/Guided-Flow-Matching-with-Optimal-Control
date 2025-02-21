#!/bin/bash
#SBATCH --job-name=so3_in_eu

cd rlhf_fm/peptide

source ~/.bashrc
conda activate flow

export PYTHONPATH=$(pwd):$PYTHONPATH


# python rlhf_finetune/eval_samples.py
mpirun -np $SLURM_NTASKS python rlhf_finetune/samples_left.py  --reg_rot 0 --start_data 0 --todo_data 0 --n_tasks 1 --alpha 0.95 --beta 2.0 --logdir rlhf_fm/peptide/exp_rebuttal/so3_in_eu --algorithm oc_so3_opt --debug