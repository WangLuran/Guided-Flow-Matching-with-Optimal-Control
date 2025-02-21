# OC-Flow

This code follows from [FlowGrad](https://github.com/gnobitab/FlowGrad).

## Controlling Rectified Flow on CelebA-HQ

We provide the scripts for applying OC-Flow to control the output of pre-trained Rectified Flow model on CelebA-HQ.

The pre-trained generative model can be downloaded from [Rectified Flow CelebA-HQ](https://drive.google.com/file/d/1ryhuJGz75S35GEdWDLiq4XFrsbwPdHnF/view?usp=sharing) 
Just put it in ``` ./ ```

### Dependencies
The following packages are required,

```
torch, numpy, lpips, clip, ml_collections, absl-py 
```

We also provide a build_env.sh script to install the dependencies.

### Run

We provide a demo image ``` ./demo/celeba.jpg ``` for running our model.

```
python main_data.py
```

### Dataset

The full Celeba-hq-1024 dataset can be downloaded from [kaggle celeba-hq dataset](https://www.kaggle.com/datasets/lamsimon/celebahq)
