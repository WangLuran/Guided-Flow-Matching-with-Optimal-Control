# Training Free Optimal Control Flow (OC-FLOW)

This is the official repo for the ICLR 2025 paper *Training Free Optimal Control Flow* by Luran Wang, Chaoran Cheng, Yizhen Liao, Yanru Qu, Ge liu. The paper is available at [arXiv](https://arxiv.org/abs/2410.18070).

<p align="center">
  <img src="pictures/iclr_2025_overview.png" alt="Triangle plot" width="100%"/>
</p>

## Introduction
We introduce *OC-Flow* as a general and theoretically grounded framework for guided flow matching. By formulating gradient guidance within the optimal control framework, we present the first training-free approach with proven convergence in both Euclidean and SO(3) spaces. OC-Flow demonstrates superior performance in extensive experiments on text-guided image manipulation, conditional molecule generation, and peptide backbone design. Check out our paper for more details!

## Requirements

To apply OC-Flow to control the output of pre-trained Rectified Flow model on CelebA-HQ, the following packages are required,

```
torch, numpy, lpips, clip, ml_collections, absl-py 
```

We also provide a build_env.sh script to install the dependencies.

## Datasets and Pretrained Model Weights

### Celeba-hq-1024

The Celeba-hq-1024 dataset is a high-quality version of CelebA that consists of 30,000 images at 1024×1024 resolution.The full Celeba-hq-1024 dataset can be downloaded from [kaggle celeba-hq dataset](https://www.kaggle.com/datasets/lamsimon/celebahq).

The pre-trained generative model can be downloaded from [Rectified Flow CelebA-HQ](https://drive.google.com/file/d/1ryhuJGz75S35GEdWDLiq4XFrsbwPdHnF/view?usp=sharing) 
Just put it in ``` ./ ```

### QM9

The QM9 dataset contains 133885 small molecules consisting of C, H, O, N, and F. The QM9 electron density dataset was built by Jørgensen et al. ([paper](https://www.nature.com/articles/s41524-022-00863-y)) and was publicly available via [Figshare](https://data.dtu.dk/articles/dataset/QM9_Charge_Densities_and_Energies_Calculated_with_VASP/16794500). 

Each tarball needs to be extracted, but the inner lz4 compression should be kept. We provided code to read the compressed lz4 file.

### Peptide Design

The data and pretrained model weights used for peptide design tasks are provided in [PepFlow](https://github.com/Ced3-han/PepFlowww)
Stay tuned for updates and feel free to reach out for collaboration or discussions!

## Usage
### iamge

We provide a demo image ``` ./demo/celeba.jpg ``` for running our model to control the output of pre-trained Rectified Flow model on CelebA-HQ.

```
python main_data.py
```

### QM9

### Peptide Design

To reproduce the experiments in the paper, run the following command:

```
sbatch peptide/submit_rlhf_fm.py
```



## Reference
If you find this repo useful, please consider citing our paper:
```bibtex
@article{wang2024training,
  title={Training Free Guided Flow Matching with Optimal Control},
  author={Wang, Luran and Cheng, Chaoran and Liao, Yizhen and Qu, Yanru and Liu, Ge},
  journal={arXiv preprint arXiv:2410.18070},
  year={2024}
}
