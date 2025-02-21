import gc
import io
import os
import time

import numpy as np
import logging
import lpips

# Keep the import below for registering all model definitions
from RectifiedFlow.models import ddpm, ncsnv2, ncsnpp
from RectifiedFlow.models import utils as mutils
from RectifiedFlow.models.ema import ExponentialMovingAverage
from absl import flags
import torch
from torchvision.utils import make_grid, save_image
from RectifiedFlow.utils import save_checkpoint, restore_checkpoint
import RectifiedFlow.datasets as datasets

from RectifiedFlow.models.utils import get_model_fn
from RectifiedFlow.models import utils as mutils

from .flowgrad_utils import get_img, embed_to_latent, clip_semantic_loss, save_img, generate_traj, flowgrad_optimization 

FLAGS = flags.FLAGS

def flowgrad_edit(config, text_prompt, alpha, model_path, image_path, output_folder="output"):
  # Create data normalizer and its inverse
  scaler = datasets.get_data_scaler(config)
  inverse_scaler = datasets.get_data_inverse_scaler(config)

  # Initialize model
  score_model = mutils.create_model(config)
  ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
  state = dict(model=score_model, ema=ema, step=0)

  state = restore_checkpoint(model_path, state, device=config.device)
  ema.copy_to(score_model.parameters())

  model_fn = mutils.get_model_fn(score_model, train=False)

  # Load the image to edit
  original_img = get_img(image_path)  
  
  log_folder = os.path.join(output_folder, 'figs')
  print('Images will be saved to:', log_folder)
  if not os.path.exists(log_folder): os.makedirs(log_folder)
  save_img(original_img, path=os.path.join(log_folder, 'original.png'))

  # Get latent code of the image and save reconstruction
  original_img = original_img.to(config.device)
  clip_loss = clip_semantic_loss(text_prompt, original_img, config.device, alpha=alpha, inverse_scaler=inverse_scaler)  

  t_s = time.time()
  latent = embed_to_latent(model_fn, scaler(original_img))
  traj = generate_traj(model_fn, latent, N=100)
  save_img(inverse_scaler(traj[-1]), path=os.path.join(log_folder, 'reconstruct.png'))
  print('Finished getting latent code and reconstruction; image saved.')
  
  # Edit according to text prompt
  u_ind = [i for i in range(100)]
  opt_u = flowgrad_optimization(latent, u_ind, model_fn, generate_traj, N=100, L_N=clip_loss.L_N, u_init=None,  number_of_iterations=10, straightness_threshold=1e-3, lr=10.0) 

  traj = generate_traj(model_fn, latent, u=opt_u, N=100)
   
  print('Total time:', time.time() - t_s)
  save_img(inverse_scaler(traj[-1]), path=os.path.join(log_folder, 'optimized.png'))
  print('Finished Editting; images saved.')


from tqdm import tqdm


def flowgrad_edit_batch(config, model_path, image_paths, text_prompt, output_dir):
  alpha = 0.7

  # Create data normalizer and its inverse
  scaler = datasets.get_data_scaler(config)
  inverse_scaler = datasets.get_data_inverse_scaler(config)

  # Initialize model
  score_model = mutils.create_model(config)
  ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
  state = dict(model=score_model, ema=ema, step=0)

  state = restore_checkpoint(model_path, state, device=config.device)
  ema.copy_to(score_model.parameters())

  model_fn = mutils.get_model_fn(score_model, train=False)
  
  N = 100
  batch_size = 1

  metrics = {}

  for img_path in tqdm(image_paths):
    target_dir = f'examples/{output_dir}'
    if img_path.startswith('examples/original'):
      opt_img_path = img_path.replace('examples/original', target_dir)
    else:
      opt_img_path = None

    # Load the image to edit
    image = get_img(img_path)  
    
    # Get latent code of the image and save reconstruction
    original_img = image.to(config.device)
    clip_loss = clip_semantic_loss(text_prompt, original_img, config.device, alpha=alpha, inverse_scaler=inverse_scaler)  

    t_s = time.time()
    latent = embed_to_latent(model_fn, scaler(original_img))
    traj = generate_traj(model_fn, latent, N=N)
    print('Finished getting latent code and reconstruction; image saved.')

    # Edit according to text prompt
    u_ind = [i for i in range(N)]
    opt_u = flowgrad_optimization(latent, u_ind, model_fn, generate_traj, N=N, L_N=clip_loss.L_N, u_init=None,  number_of_iterations=15, straightness_threshold=None, lr=10.0) 

    traj_opt = generate_traj(model_fn, latent, u=opt_u, N=N)
    
    if opt_img_path is not None:
      save_img(inverse_scaler(traj_opt[-1]), path=opt_img_path)

    with torch.no_grad():
      clip_loss_1 = clip_semantic_loss(text_prompt, original_img, config.device, alpha=1., inverse_scaler=inverse_scaler)  
      # id_loss = IDLoss(device=config.device)

      lpips_f = lpips.LPIPS(net='alex').to(config.device) # or 'vgg', 'squeeze'

      clip_loss = clip_loss_1.L_N(traj_opt[-1]).item()
      lpips_score = lpips_f(traj_opt[-1], traj[-1]).item()
      print(f'clip loss: {clip_loss:.4f}, lpips score: {lpips_score:.4f}, total time: {time.time() - t_s:.4f} s')

    metrics[opt_img_path] = {
      'clip_loss': clip_loss,
      'lpips_score': lpips_score,
    }

  torch.save(metrics, f'{target_dir}/metrics.pt')
  return metrics
