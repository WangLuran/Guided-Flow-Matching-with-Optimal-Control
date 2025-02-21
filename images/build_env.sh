mamba create -n flowgrad python=3.10

mamba activate flowgrad

mamba install numpy=1.26.4 pytorch=2.1.1 torchvision=0.16.1 pytorch-cuda=11.8 -c pytorch -c nvidia

mamba install ninja=1.12.1 absl-py=2.1.0

mamba install lpips=0.1.3 ml-collections=0.1.1 openai-clip=1.0.1

mamba install numpy=1.26.4 pytorch=2.4.1 torchvision=0.19.1 pytorch-cuda=12.4 lpips=0.1.3 openai-clip=1.0.1 ml-collections=0.1.1 absl-py=2.1.0 ninja=1.12.1 -c pytorch -c nvidia
