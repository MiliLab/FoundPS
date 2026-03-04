import os
import numpy as np
import random
import argparse

import torch

from model import create_model 
from train import Trainer
from test import Tester
 
def set_seed(SEED):
    # initialize random seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

def train_ddp_accelerate(): # args
    datafolder = '' 
    model = create_model()
    PS_Trainer = Trainer(
        model,
        datafolder,
        train_num_steps = 500001,
        save_and_sample_every = 5000,
        batch_size = 6,
        image_size = 256,
        results_folder = './meta/', 
    )  
    PS_Trainer.train() 
    print('Procedure Termination: (Finished)')


def test_ddp_accelerate():
    datafolder = ''  
    model = create_model() 
    PS_Tester = Tester(
        model,
        datafolder,
        train_num_steps = 500001,
        save_and_sample_every = 5000,
        batch_size = 8,
        image_size = 256,
        results_folder = './meta/', 
    )    
    # model_index = os.listdir('./meta')[0].split('-')[-1]
    # print('Loading Model : ', int(model_index))
    PS_Tester.load()
    PS_Tester.test() 
    print('Procedure Termination: (Finished)')

def parse_args():
    parser = argparse.ArgumentParser(description='FoundPS Train/Test Launcher')

    # 核心参数
    parser.add_argument(
        '--mode',
        type=str,
        default=None,
        choices=['train', 'test'],
        help='Run mode: train or test'
    )

    return parser.parse_args()


if __name__ == '__main__':  
    args = parse_args()
    set_seed(0)
    print('Procedure Running: FoundPS')
    if args.mode == 'train':
        train_ddp_accelerate()
    elif args.mode == 'test':
        test_ddp_accelerate()
    else:
        raise ValueError(f'Unknown mode: {args.mode}')   