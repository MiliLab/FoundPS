import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T, utils 

from PIL import Image
from torch import nn
import imageio
import cv2
import numpy as np
import random
import json
 
def exists(x):
    return x is not None

def cycle(dl):
    while True:
        for data in dl:
            yield data

def default(val, d):
    if exists(val) and (val is not None):
        return val
    return d() if callable(d) else d
 
class train_dataset(Dataset):
    def __init__(self, root_dir, cls_folder = None, band_folder = None, mode = 'train', image_size = 256):
        super().__init__() 
        self.root_dir = root_dir
        self.cls_folder_list = os.listdir(root_dir)
        assert mode in ['train','test','meta']
        assert cls_folder in self.cls_folder_list, f'Wrong CLS_FOLDER'
        assert band_folder in os.listdir(os.path.join(root_dir,cls_folder)), f'Wrong BAND_FOLDER'
        self.cls_folder = cls_folder
        self.transform = T.ToTensor()
        self.image_size = image_size

        self.meta_path = os.path.join(root_dir,cls_folder,band_folder,'{}.json'.format(mode))

        with open(self.meta_path, "r", encoding="utf-8") as f:
            self.meta_info = f.readlines()

        self.meta_list = [json.loads(x) for x in self.meta_info if x.strip()] 

        self.transform = T.Compose([T.ToTensor()])
 
    def __len__(self):
        assert len(self.meta_list) > 0, f"meta_list is empty, check file: {self.meta_path}"
        return len(self.meta_list)
    
    def __getitem__(self, item): 
        image_pan_path = self.meta_list[item]['pan']
        image_ms_path = self.meta_list[item]['ms']
        image_ms_label_path = self.meta_list[item]['ms_label']
        image_pan_label_path = self.meta_list[item]['pan_label']

        if self.cls_folder == 'landsat7':
            image_pan = self.load_image(image_pan_path, data_range = 255.0)
            image_ms = self.load_image(image_ms_path, data_range = 255.0) 
            image_ms_label = self.load_image(image_ms_label_path, data_range = 255.0)
            image_pan_label = self.load_image(image_pan_label_path, data_range = 255.0)
        elif self.cls_folder == 'cresda':
            image_pan = self.load_image(image_pan_path, data_range = 4095.0)
            image_ms = self.load_image(image_ms_path, data_range =4095.0) 
            image_ms_label = self.load_image(image_ms_label_path, data_range = 4095.0)
            image_pan_label = self.load_image(image_pan_label_path, data_range = 4095.0) 
        elif self.cls_folder == 'maxar':
            image_pan = self.load_image(image_pan_path, data_range = 10000.0)
            image_ms = self.load_image(image_ms_path, data_range = 10000.0) 
            image_ms_label = self.load_image(image_ms_label_path, data_range = 10000.0)
            image_pan_label = self.load_image(image_pan_label_path, data_range = 10000.0) 
        elif self.cls_folder == 'landsat8':
            image_pan = self.load_image(image_pan_path, data_range = 65535.0)
            image_ms = self.load_image(image_ms_path, data_range = 65535.0) 
            image_ms_label = self.load_image(image_ms_label_path, data_range = 65535.0)
            image_pan_label = self.load_image(image_pan_label_path, data_range = 65535.0) 
        elif self.cls_folder == 'landsat9':
            image_pan = self.load_image(image_pan_path, data_range = 65535.0)
            image_ms = self.load_image(image_ms_path, data_range = 65535.0) 
            image_ms_label = self.load_image(image_ms_label_path, data_range = 65535.0)
            image_pan_label = self.load_image(image_pan_label_path, data_range = 65535.0) 
        elif self.cls_folder == 'others':
            image_pan = self.load_image(image_pan_path, data_range = 255.0)
            image_ms = self.load_image(image_ms_path, data_range = 255.0) 
            image_ms_label = self.load_image(image_ms_label_path, data_range = 255.0)
            image_pan_label = self.load_image(image_pan_label_path, data_range = 255.0) 
        else:
            exit(0)

        assert (image_pan.shape[0] / image_ms.shape[0]) == (image_pan_label.shape[0] / image_ms_label.shape[0]), 'image sizes are not matched'

        image_pan,image_ms,image_ms_label,image_pan_label = self.random_crop_size(image_pan,image_ms,image_ms_label,image_pan_label,self.image_size)
        
        if self.transform is not None:
            image_pan = self.transform(image_pan)
            image_ms = self.transform(image_ms)
            image_pan_label = self.transform(image_pan_label)
            image_ms_label = self.transform(image_ms_label)
            pass

        return image_ms_path, image_pan, image_ms, image_pan_label, image_ms_label

    def load_image(self,path,data_range):
        tmp = np.clip(imageio.imread(path),0,data_range).astype('float32') / data_range
        return tmp

    def random_crop_size(self,image_pan,image_ms,image_ms_label,image_pan_label,crop_size):  
        ratio = image_pan_label.shape[0] / image_ms_label.shape[0]
        h,w = image_pan.shape
        h_start,w_start = np.random.randint(0,h-crop_size+1),np.random.randint(0,w-crop_size+1)
        image_pan_crop = image_pan[h_start:h_start+crop_size,w_start:w_start+crop_size]
        image_ms_crop = image_ms[int(h_start//ratio):int(h_start//ratio+crop_size//ratio),int(w_start//ratio):int(w_start//ratio+crop_size//ratio),:]
        image_ms_label_crop = image_ms_label[h_start:h_start+crop_size,w_start:w_start+crop_size,:]
        image_pan_label_crop = image_pan_label[int(h_start*ratio):int(h_start*ratio+crop_size*ratio),int(w_start*ratio):int(w_start*ratio+crop_size*ratio)]
        
        return image_pan_crop,image_ms_crop,image_ms_label_crop,image_pan_label_crop


class test_dataset(Dataset):
    def __init__(self, root_dir, cls_folder = None, band_folder = None, image_size = 256):
        super().__init__() 
        self.root_dir = root_dir
        self.cls_folder_list = os.listdir(root_dir)
        assert cls_folder in self.cls_folder_list, f'Wrong CLS_FOLDER'
        assert band_folder in os.listdir(os.path.join(root_dir,cls_folder)), f'Wrong BAND_FOLDER'
        self.cls_folder = cls_folder
        self.transform = T.ToTensor()
        self.image_size = image_size

        self.meta_path = os.path.join(root_dir,cls_folder,band_folder,'test.json')

        with open(self.meta_path, "r", encoding="utf-8") as f:
            self.meta_info = f.readlines()

        self.meta_list = [json.loads(x) for x in self.meta_info if x.strip()] 

        self.transform = T.Compose([T.ToTensor()])
 
    def __len__(self):
        assert len(self.meta_list) > 0, f"meta_list is empty, check file: {self.meta_path}"
        return len(self.meta_list)
    
    def __getitem__(self, item): 
        image_pan_path = self.meta_list[item]['pan']
        image_ms_path = self.meta_list[item]['ms']
        image_ms_label_path = self.meta_list[item]['ms_label']
        image_pan_label_path = self.meta_list[item]['pan_label']

        if self.cls_folder in ['landsat7','SegGF','QuickBird']:
            image_pan = self.load_image(image_pan_path, data_range = 255.0)
            image_ms = self.load_image(image_ms_path, data_range = 255.0) 
            image_ms_label = self.load_image(image_ms_label_path, data_range = 255.0)
            image_pan_label = self.load_image(image_pan_label_path, data_range = 255.0)
        elif self.cls_folder == 'cresda':
            image_pan = self.load_image(image_pan_path, data_range = 4095.0)
            image_ms = self.load_image(image_ms_path, data_range = 4095.0) 
            image_ms_label = self.load_image(image_ms_label_path, data_range = 4095.0)
            image_pan_label = self.load_image(image_pan_label_path, data_range = 4095.0) 
        elif self.cls_folder == 'maxar':
            image_pan = self.load_image(image_pan_path, data_range = 10000.0)
            image_ms = self.load_image(image_ms_path, data_range = 10000.0) 
            image_ms_label = self.load_image(image_ms_label_path, data_range = 10000.0)
            image_pan_label = self.load_image(image_pan_label_path, data_range = 10000.0) 
        elif self.cls_folder == 'landsat8':
            image_pan = self.load_image(image_pan_path, data_range = 65535.0)
            image_ms = self.load_image(image_ms_path, data_range = 65535.0) 
            image_ms_label = self.load_image(image_ms_label_path, data_range = 65535.0)
            image_pan_label = self.load_image(image_pan_label_path, data_range = 65535.0) 
        elif self.cls_folder == 'landsat9':
            image_pan = imageio.imread(image_pan_path).astype('float32') / 65535.0
            image_ms = imageio.imread(image_ms_path).astype('float32') / 65535.0
            image_ms_label = imageio.imread(image_ms_label_path).astype('float32') / 65535.0
            image_pan_label = imageio.imread(image_pan_label_path).astype('float32') / 65535.0
        elif self.cls_folder == 'others':
            image_pan = self.load_image(image_pan_path, data_range = 255.0)
            image_ms = self.load_image(image_ms_path, data_range = 255.0) 
            image_ms_label = self.load_image(image_ms_label_path, data_range = 255.0)
            image_pan_label = self.load_image(image_pan_label_path, data_range = 255.0) 
        else:
            exit(0)
            
        assert (image_pan.shape[0] / image_ms.shape[0]) == (image_pan_label.shape[0] / image_ms_label.shape[0]), 'image sizes are not matched'

        if self.transform is not None:
            image_pan = self.transform(image_pan)
            image_ms = self.transform(image_ms)
            image_pan_label = self.transform(image_pan_label)
            image_ms_label = self.transform(image_ms_label)
            pass

        return image_ms_path, image_pan, image_ms, image_pan_label, image_ms_label

    def load_image(self,path,data_range):
        tmp = np.clip(imageio.imread(path),0,data_range).astype('float32') / data_range
        return tmp

if __name__ == '__main__':
    print('Hello World')
    root_dir = ''
    cls_folders = ['cresda','maxar','landsat8','landsat9']
    ds = train_dataset(root_dir, cls_folder = 'cresda', band_folder = '4', image_size = 256)
    image_ms_label_file, image_pan, image_ms, image_pan_label, image_ms_label = ds[1]
    print(image_pan.shape, image_ms.shape, image_pan_label.shape, image_ms_label.shape)
    print(image_ms_label_file, image_pan.min(), image_pan.max())
    print(image_ms.min(), image_ms.max())
    print(image_pan_label.min(),image_pan_label.max()) 
    print(image_ms_label.min(),image_ms_label.max())

    ds = train_dataset(root_dir, cls_folder = 'maxar', band_folder = '4', image_size = 256)
    image_ms_label_file, image_pan, image_ms, image_pan_label, image_ms_label = ds[1]
    print(image_pan.shape, image_ms.shape, image_pan_label.shape, image_ms_label.shape)
    print(image_ms_label_file, image_pan.min(), image_pan.max())
    print(image_ms.min(), image_ms.max())
    print(image_pan_label.min(),image_pan_label.max()) 
    print(image_ms_label.min(),image_ms_label.max())
    
    ds = train_dataset(root_dir, cls_folder = 'maxar', band_folder = '8', image_size = 256)
    image_ms_label_file, image_pan, image_ms, image_pan_label, image_ms_label = ds[951]
    print(image_pan.shape, image_ms.shape, image_pan_label.shape, image_ms_label.shape)
    print(image_ms_label_file, image_pan.min(), image_pan.max())
    print(image_ms.min(), image_ms.max())
    print(image_pan_label.min(),image_pan_label.max()) 
    print(image_ms_label.min(),image_ms_label.max())

    ds = train_dataset(root_dir, cls_folder = 'landsat8', band_folder = '10', image_size = 256)
    image_ms_label_file, image_pan, image_ms, image_pan_label, image_ms_label = ds[951]
    print(image_pan.shape, image_ms.shape, image_pan_label.shape, image_ms_label.shape)
    print(image_ms_label_file, image_pan.min(), image_pan.max())
    print(image_ms.min(), image_ms.max())
    print(image_pan_label.min(),image_pan_label.max()) 
    print(image_ms_label.min(),image_ms_label.max())

    ds = train_dataset(root_dir, cls_folder = 'landsat9', band_folder = '10', image_size = 256)
    image_ms_label_file, image_pan, image_ms, image_pan_label, image_ms_label = ds[951]
    print(image_pan.shape, image_ms.shape, image_pan_label.shape, image_ms_label.shape)
    print(image_ms_label_file, image_pan.min(), image_pan.max())
    print(image_ms.min(), image_ms.max())
    print(image_pan_label.min(),image_pan_label.max()) 
    print(image_ms_label.min(),image_ms_label.max())


    print('Finished')