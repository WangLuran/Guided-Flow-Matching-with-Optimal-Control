# coding=utf-8

from utils import run_lib_flowgrad_oc
from absl import app
from absl import flags
from ml_collections.config_flags import config_flags
import logging
import os
import torchvision.transforms as transforms
from torchvision.datasets import CelebA
from torch.utils.data import DataLoader, SubsetRandomSampler
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from torchvision import models
import numpy as np
from torchvision import datasets, transforms
import imageio
import json

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file("config", 'RectifiedFlow/configs/celeba_hq_pytorch_rf_gaussian.py', "Rectified Flow Model configuration.", lock_config=True)
flags.DEFINE_string('method', 'flowgrad', '[flowgrad, ocfm]')
flags.DEFINE_integer("batch_size", 1, "batch size")
flags.DEFINE_integer("index", 0, "position of samples")
text_prompts = ['A photo of an old face.','A photo of a sad face.','A photo of a smiling face.','A photo of an angry face.','A photo of a face with curly hair.']

# import glob

# all_image_paths = sorted(glob.glob('data/data_celeba_hq_1024/*.jpg'))
# print(len(all_image_paths))
# print(all_image_paths[:10])
# image_paths = all_image_paths[:10]

image_paths = ['./demo/celeba.jpg']

alpha = 0.7
lr = 1
model_path = './checkpoint_10.pth'


def get_img(path=None):
    img = imageio.imread(path) ### 4-no expression
    print(img.shape)
    img = img / 255.
    img = img[np.newaxis, :, :, :]
    img = img.transpose(0, 3, 1, 2)
    print('read image from:', path, 'img range:', img.min(), img.max())
    img = torch.tensor(img).float()
    img = torch.nn.functional.interpolate(img, size=256)

    return img

# Path to your CelebA images folder
image_dir = 'data_celeba_hq_1024'

class CelebADataset(Dataset):
    def __init__(self, img_dir):
        self.img_dir = img_dir
        self.image_files = os.listdir(img_dir)  # Get list of image filenames

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.image_files[idx])
        image = get_img(img_path)  # Load image
        
        return image

# Create the dataset and DataLoader
# celeba_dataset = CelebADataset(image_dir)

# print('data finish')

# Set random seed for reproducibility
np.random.seed(42)


def main(argv):

  output_dirs = ['ocfm/old', 'ocfm/sad', 'ocfm/smile', 'ocfm/angry', 'ocfm/curly']

  prompt = text_prompts[2]
  output_dir = output_dirs[2]

  metrics = run_lib_flowgrad_oc.flowgrad_edit_batch(FLAGS.config, model_path, image_paths, prompt, output_dir)

  print(metrics)


if __name__ == "__main__":
  app.run(main)

