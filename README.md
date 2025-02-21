# Training Free Optimal Control Flow (OC-FLOW)

This is the official repo for the ICLR 2025 paper *Training Free Optimal Control Flow* by Luran Wang, Chaoran Cheng, Yizhen Liao, Yanru Qu, Ge liu. The paper is available at [arXiv](https://arxiv.org/abs/2410.18070).

<p align="center">
  <img src="pictures/iclr_2025_overview.png" alt="Triangle plot" width="100%"/>
</p>

## Introduction
We introduce *OC-Flow* as a general and theoretically grounded framework for guided flow matching. By formulating gradient guidance within the optimal control framework, we present the first training-free approach with proven convergence in both Euclidean and SO(3) spaces. OC-Flow demonstrates superior performance in extensive experiments on text-guided image manipulation, conditional molecule generation, and peptide backbone design. Check out our paper for more details!

## Installation
To apply OC-Flow to control the output of pre-trained Rectified Flow model on CelebA-HQ, the following packages are required,

```
torch, numpy, lpips, clip, ml_collections, absl-py 
```

We also provide a build_env.sh script to install the dependencies.

The pre-trained generative model can be downloaded from [Rectified Flow CelebA-HQ](https://drive.google.com/file/d/1ryhuJGz75S35GEdWDLiq4XFrsbwPdHnF/view?usp=sharing) 
Just put it in ``` ./ ```

The full Celeba-hq-1024 dataset can be downloaded from [kaggle celeba-hq dataset](https://www.kaggle.com/datasets/lamsimon/celebahq)

### Usage

We provide a demo image ``` ./demo/celeba.jpg ``` for running our model to control the output of pre-trained Rectified Flow model on CelebA-HQ.

```
python main_data.py
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
