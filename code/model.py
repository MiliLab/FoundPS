import torch
import torch.nn as nn 
import torch.nn.functional as F

from foundps import FoundPS


def create_tiny_model():
    return None

def create_small_model():
    return None

def create_model():
    return FoundPS(        
        latent_dim = 64,  # hidden_state N dim  == Unet Input Channel 
        num_experts = 16, # num of experts 
        dim = 64, #LatentUNet
        dim_mults=(1, 2, 2, 4), 
        objective = 'pred_x_start', 
        sampling_type = 'pred_x_start',
        timesteps=1000, 
        sampling_timesteps=3,  
        )


def create_large_model():
    return None


def create_hyper_model():
    return None