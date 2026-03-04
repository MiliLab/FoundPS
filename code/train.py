import os
import numpy as np 
import time
import imageio
import json
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim import Adam
from torchvision import transforms as T, utils
from torch.utils.data import DataLoader
from ema_pytorch import EMA
from tqdm import tqdm

import accelerate
from accelerate import Accelerator, DistributedDataParallelKwargs

from datasets_setting import train_dataset,test_dataset 
from evaluator import Evaluator
from visualization import RSGenerate


def cycle(dl): 
    while True: 
        for data in dl:
            yield data

def divisible_by(numer, denom):
    return (numer % denom) == 0

def create_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

def create_empty_json(json_path):
    with open(json_path, 'w') as file:
        pass
 
def remove_json(json_path):
    os.remove(json_path)

def write_json(json_path,item):
    with open(json_path, 'a+', encoding='utf-8') as f:
        line = json.dumps(item)
        f.write(line+'\n')

def readline_json(json_path,key=None):
    data = []
    with open(json_path, 'r') as f:
        items = f.readlines()
    file_flag = []
    if key is not None:
        for item in items:
            file_name = json.loads(item)['file_path']
            if file_name not in file_flag:
                file_flag.append(file_name)
                data.append(json.loads(item)[key])
        return np.asarray(data).mean()#,len(file_flag)
    else:
        for item in items:
            data.append(json.loads(item))
        return data


class Trainer(object):
    def __init__(
        self,
        model,
        datafolder = '',
        train_num_steps = 500001,
        save_and_sample_every = 5000,
        batch_size = 4,
        image_size = 256,
        results_folder = './meta/', 
        *, 
        train_lr = 3e-4,
        ema_update_every = 1,
        ema_decay = 0.995,
        adam_betas = (0.9, 0.99), 
        split_batches = True, 
        max_grad_norm = 1.,
    ):
        super().__init__()

        self.accelerator = Accelerator(
            split_batches = split_batches,
            kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)]
        )

        self.model = model  
        self.save_and_sample_every = save_and_sample_every
        self.batch_size = batch_size 
        self.image_size = image_size
        self.max_grad_norm = max_grad_norm 
        self.train_num_steps = train_num_steps
        self.step = 0

        self.train_folder = datafolder

        self.ds_cresda_4 = train_dataset(self.train_folder, cls_folder = 'cresda', band_folder = '4', mode = 'train', image_size = 256)
        self.dl_cresda_4 = cycle(self.accelerator.prepare(DataLoader(self.ds_cresda_4, self.batch_size)))

        self.ds_maxar_4 = train_dataset(self.train_folder, cls_folder = 'maxar', band_folder = '4', mode = 'train', image_size = 256)
        self.dl_maxar_4 = cycle(self.accelerator.prepare(DataLoader(self.ds_maxar_4, self.batch_size)))

        self.ds_maxar_8 = train_dataset(self.train_folder, cls_folder = 'maxar', band_folder = '8', mode = 'train', image_size = 256)
        self.dl_maxar_8 = cycle(self.accelerator.prepare(DataLoader(self.ds_maxar_8, self.batch_size)))

        self.ds_landsat7_7 = train_dataset(self.train_folder, cls_folder = 'landsat7', band_folder = '7', mode = 'train', image_size = 256)
        self.dl_landsat7_7 = cycle(self.accelerator.prepare(DataLoader(self.ds_landsat7_7, self.batch_size)))

        self.ds_landsat8_10 = train_dataset(self.train_folder, cls_folder = 'landsat8', band_folder = '10', mode = 'train', image_size = 256)
        self.dl_landsat8_10 = cycle(self.accelerator.prepare(DataLoader(self.ds_landsat8_10, self.batch_size)))

        self.ds_landsat9_10 = train_dataset(self.train_folder, cls_folder = 'landsat9', band_folder = '10', mode = 'train', image_size = 256)
        self.dl_landsat9_10 = cycle(self.accelerator.prepare(DataLoader(self.ds_landsat9_10, self.batch_size)))

        self.dataloader_set = [
            self.dl_cresda_4, 
            self.dl_maxar_4,  
            self.dl_maxar_8,     
            self.dl_landsat7_7,   
            self.dl_landsat8_10,  
            self.dl_landsat9_10
        ]

        if self.accelerator.is_main_process:
            self.accelerator.print('Training Samplies : ( cresda_4 :{})'.format(len(self.ds_cresda_4)))
            self.accelerator.print('Training Samplies : ( maxar_4  :{})'.format(len(self.ds_maxar_4)))
            self.accelerator.print('Training Samplies : ( maxar_8  :{})'.format(len(self.ds_maxar_8)))
            self.accelerator.print('Training Samplies : ( landsat7_7 :{})'.format(len(self.ds_landsat7_7)))
            self.accelerator.print('Training Samplies : ( landsat8_10 :{})'.format(len(self.ds_landsat8_10)))
            self.accelerator.print('Training Samplies : ( landsat9_10  :{})'.format(len(self.ds_landsat9_10)))
        
            
        self.opt = Adam(model.parameters(), lr = train_lr, betas = adam_betas)
        self.ema = EMA(model, beta = ema_decay, update_every = ema_update_every)
        self.ema.to(self.device)
        
        self.model, self.opt = self.accelerator.prepare(self.model, self.opt) 
        self.evaluator = Evaluator()

        self.results_folder = results_folder
        if self.accelerator.is_main_process:
            create_folder(self.results_folder)

    @property
    def device(self):
        return self.accelerator.device

    def save(self, milestone = None):
        if not self.accelerator.is_local_main_process:
            return
        data = {
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.model),
            'opt': self.opt.state_dict(),
            'ema': self.ema.state_dict(),
        }
        checkpoint_save_path = os.path.join(self.results_folder,f'model-{milestone}')
        if not os.path.exists(checkpoint_save_path):
            os.makedirs(checkpoint_save_path)
        torch.save(data, checkpoint_save_path + '/' +  f'model-{milestone}.pt')

    def load(self, milestone= None, assess = False, from_scratch = False):
        accelerator = self.accelerator
        device = accelerator.device
        checkpoint_save_path = os.path.join(self.results_folder,f'model-{milestone}')
        data = torch.load(str(checkpoint_save_path + '/' + f'model-{milestone}.pt'), map_location=device)
        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data['model'])
        if from_scratch:
            self.step = 0
        else:
            if not assess:
                self.step = data['step'] + 1
            else:
                self.step = data['step']

        self.opt.load_state_dict(data['opt']) 
        self.ema.load_state_dict(data["ema"])
        print('Load ',milestone,'Assess ',assess,'From Start ',from_scratch,'Iter ',self.step)

    def get_dataloader(self): 
        train_iter = self.dataloader_set.pop(0) 
        self.dataloader_set.append(train_iter)
        return train_iter

    def train(self):
        accelerator = self.accelerator
        device = accelerator.device
        track_metric_json_path = os.path.join(self.results_folder,'metric.json') 
        if self.accelerator.is_main_process: 
            if not os.path.exists(track_metric_json_path):
                create_empty_json(track_metric_json_path)
        with tqdm(initial=self.step, total=self.train_num_steps, disable=not accelerator.is_main_process) as pbar:
            while self.step < self.train_num_steps:
                self.model.train()
                train_iter, valid_iter = self.get_dataloader()
                file_name, image_pan, image_ms, image_pan_label, image_ms_label = next(train_iter)

                with self.accelerator.autocast():
                    self.opt.zero_grad() 
                    pixel_loss,latent_loss,aux_loss = self.model(pan = image_pan.to(device), ms = image_ms.to(device), gt = image_ms_label.to(device))
                    loss = pixel_loss + latent_loss + aux_loss
                    self.accelerator.backward(loss)
                    accelerator.wait_for_everyone()
                    accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.opt.step() 
                    self.ema.update()
                    accelerator.wait_for_everyone()
                    pbar.set_description(f'loss -> {loss.item():.4f} : pixel -> {pixel_loss.item():.4f} latent -> {latent_loss.item():.4f}  auxiluary  -> {aux_loss.item():.4f} ')

                accelerator.wait_for_everyone() 

                if self.accelerator.is_main_process:
                    if self.step != 0 and divisible_by(self.step, self.save_and_sample_every):  
                        accelerator.print('save model checkpoint')  
                        self.save(self.step)  

                accelerator.wait_for_everyone()      
                self.step += 1
                pbar.update(1)
                
        accelerator.print('Training complete')

    def tf2np(self,image_tf):
        n,c,h,w = image_tf.size()
        assert n == 1
        if c == 1:
            image_np = image_tf.squeeze(0).squeeze(0).detach().cpu().numpy()
        else:
            image_np = image_tf.squeeze(0).permute(1,2,0).detach().cpu().numpy()
        
        return image_np

    def tf2img(self,image_tf, data_range):
        image_np = self.tf2np(torch.clamp(image_tf,min=0.,max=1.))
        image_np = (image_np * data_range).astype(np.uint16)
        return image_np
  

if __name__  == '__main__':
    print('Hello World')

    print('Finished')