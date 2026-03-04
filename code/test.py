import os
import numpy as np 
import time
import imageio.v3 as imageio
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


class Tester(object):
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

        self.model = model  
        self.save_and_sample_every = save_and_sample_every
        self.batch_size = batch_size 
        self.image_size = image_size
        self.max_grad_norm = max_grad_norm 
        self.train_num_steps = train_num_steps
        self.step = 0
        
        self.test_folder = datafolder

        self.ds_cresda_4_test = test_dataset(self.test_folder, cls_folder = 'cresda', band_folder = '4', image_size = 256)
        self.dl_cresda_4_test = self.accelerator.prepare(DataLoader(self.ds_cresda_4_test, batch_size = 1))

        self.ds_maxar_4_test = test_dataset(self.test_folder, cls_folder = 'maxar', band_folder = '4', image_size = 256)
        self.dl_maxar_4_test = self.accelerator.prepare(DataLoader(self.ds_maxar_4_test, batch_size = 1))

        self.ds_maxar_8_test = test_dataset(self.test_folder, cls_folder = 'maxar', band_folder = '8', image_size = 256)
        self.dl_maxar_8_test = self.accelerator.prepare(DataLoader(self.ds_maxar_8_test, batch_size = 1))

        self.ds_landsat7_7_test = test_dataset(self.test_folder, cls_folder = 'landsat7', band_folder = '7', image_size = 256)
        self.dl_landsat7_7_test = self.accelerator.prepare(DataLoader(self.ds_landsat7_7_test, 1))

        self.ds_landsat8_10_test = test_dataset(self.test_folder, cls_folder = 'landsat8', band_folder = '10', image_size = 256)
        self.dl_landsat8_10_test = self.accelerator.prepare(DataLoader(self.ds_landsat8_10_test, batch_size = 1))

        self.ds_landsat9_10_test = test_dataset(self.test_folder, cls_folder = 'landsat9', band_folder = '10', image_size = 256)
        self.dl_landsat9_10_test = self.accelerator.prepare(DataLoader(self.ds_landsat9_10_test, batch_size = 1)) 

        if self.accelerator.is_main_process:
            self.accelerator.print('Validation Samplies : ( cresda_4 :{})'.format(len(self.ds_cresda_4_test)))
            self.accelerator.print('Validation Samplies : ( maxar_4  :{})'.format(len(self.ds_maxar_4_test)))
            self.accelerator.print('Validation Samplies : ( maxar_8  :{})'.format(len(self.ds_maxar_8_test)))
            self.accelerator.print('Validation Samplies : ( landsat7_7 :{})'.format(len(self.ds_landsat7_7_test)))            
            self.accelerator.print('Validation Samplies : ( landsat8_10 :{})'.format(len(self.ds_landsat8_10_test)))
            self.accelerator.print('Validation Samplies : ( landsat9_10  :{})'.format(len(self.ds_landsat9_10_test)))
 
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
        train_iter, valid_iter = self.dataloader_set.pop(0) 
        self.dataloader_set.append((train_iter, valid_iter))
        return train_iter, valid_iter

    def test(self):
        accelerator = self.accelerator
        device = accelerator.device
        
        self.assess(dataloader = self.dl_cresda_4_test, sensor = 'cresda_4', BAND_NUM = 4, data_range = 4095.0, eval_range = 4095.0 )    
        self.assess(dataloader = self.dl_maxar_4_test, sensor = 'maxar_4', BAND_NUM = 4, data_range = 10000.0, eval_range = 10000.0 )    
        self.assess(dataloader = self.dl_maxar_8_test, sensor = 'maxar_8', BAND_NUM = 8, data_range = 10000.0, eval_range = 10000.0 )    
        self.assess(dataloader = self.dl_landsat7_7_test, sensor = 'landsat7', BAND_NUM = 7, data_range = 255.0, eval_range = 255.0)    
        self.assess(dataloader = self.dl_landsat8_10_test, sensor = 'landsat8',  BAND_NUM = 10, data_range = 65535.0, eval_range =65535.0)    
        self.assess(dataloader = self.dl_landsat9_10_test, sensor = 'landsat9', BAND_NUM = 10, data_range = 65535.0, eval_range =65535.0)

        accelerator.print('Testing complete')

    def assess(self,dataloader,sensor = None,BAND_NUM = None,data_range = None,eval_range = None): 
        accelerator = self.accelerator
        device = accelerator.device
        self.accelerator.wait_for_everyone() 
        if self.accelerator.is_main_process: 
            start_time = time.time()
            save_reduced_dir = os.path.join('./save_folder/',f'output',sensor,'reduced')
            save_full_dir = os.path.join('./save_folder/',f'output',sensor,'full')
            vis_reduced_dir = os.path.join('./save_folder/',f'visualization',sensor,'reduced')
            vis_full_dir = os.path.join('./save_folder/',f'visualization',sensor,'full')
            create_folder(save_reduced_dir) 
            create_folder(save_full_dir)  
            create_folder(vis_reduced_dir) 
            create_folder(vis_full_dir)  
            
        self.accelerator.wait_for_everyone()
        self.model.eval()
        save_reduced_dir = os.path.join('./save_folder/',f'output',sensor,'reduced')
        save_full_dir = os.path.join('./save_folder/',f'output',sensor,'full')
        vis_reduced_dir = os.path.join('./save_folder/',f'visualization',sensor,'reduced')
        vis_full_dir = os.path.join('./save_folder/',f'visualization',sensor,'full')

        for batch_id,batch in enumerate(dataloader):  
            file_name, image_pan, image_ms, image_pan_label, image_ms_label = batch   
                
            image_reduced = self.ema.model.sample(pan = image_pan.to(device), ms = image_ms.to(device), last=True)  
            image_full = self.ema.model.sample(pan = image_pan_label.to(device), ms = image_ms_label.to(device), last=True) 
            
            for element_id in range(1): 
                image_np_reduced = self.tf2img(image_reduced[element_id,:,:,].unsqueeze(0),data_range)   
                image_np_full = self.tf2img(image_full[element_id,:,:,].unsqueeze(0),data_range)  

                imageio.imwrite(os.path.join(save_reduced_dir,file_name[element_id].split('/')[-1][:-4]+'.tif'),image_np_reduced)
                imageio.imwrite(os.path.join(save_full_dir,file_name[element_id].split('/')[-1][:-4]+'.tif'),image_np_full)
                
                imageio.imwrite(os.path.join(vis_reduced_dir,file_name[element_id].split('/')[-1][:-4]+'.jpg'),RSGenerate(image_np_reduced,BAND_NUM))
                imageio.imwrite(os.path.join(vis_full_dir,file_name[element_id].split('/')[-1][:-4]+'.jpg'),RSGenerate(image_np_full,BAND_NUM))

                accelerator.print(batch_id, file_name[element_id])

        if self.accelerator.is_main_process: 
            end_time = time.time()
            test_time_consuming = end_time - start_time        
            self.accelerator.print('Test_time_consuming : {:.6} s'.format(test_time_consuming))

        self.accelerator.wait_for_everyone()   

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