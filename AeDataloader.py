# -*- coding: utf-8 -*-

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets

import warnings
warnings.filterwarnings("ignore")

class xIterator :
    def __init__(self, x, batch_size) : 
        self.x = x
        self.batch_size = batch_size
        self.idx = 0
    def __iter__(self):
        return self
    def __next__(self) :
        if self.idx == self.x.shape[0] :
            raise StopIteration
        
        prev = self.idx
        self.idx = min(self.idx + self.batch_size, self.x.shape[0])
        return self.x[prev : self.idx]        

class xtIterator :
    def __init__(self, x, t, batch_size) : 
        self.x = x
        self.t = t
        self.batch_size = batch_size
        self.idx = 0
    def __iter__(self):
        return self
    def __next__(self) :
        if self.idx == self.x.shape[0] :
            raise StopIteration
        
        prev = self.idx
        self.idx = min(self.idx + self.batch_size, self.x.shape[0])
        return (self.x[prev : self.idx], self.t[prev : self.idx])

class AeDataloader(Dataset) :
    def __init__(self, root, init_size, k) : 
        data_transform = transforms.Compose([transforms.ToTensor(), 
                         transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])
        cifar_data = datasets.CIFAR10(root, train=False, download=True, 
                                          transform=data_transform)
        init_loader = DataLoader(cifar_data, batch_size=init_size, shuffle=True)
        
        self.x = next(iter(init_loader))[0]
        self.t = torch.Tensor().long()
        self.x_full = self.x.shape[0] # 저장된 데이터 크기
        self.t_full = self.t.shape[0] # 저장된 label 크기
        self.k = k # augmentation 시 증가하는 데이터 수
    
    def __len__(self) :
        return self.x_full
        
    def __getitem__(self, idx) :
        return (self.x[idx], self.t[idx])
    
    def label_load(self, label_batch_size) : 
        return xIterator(self.x[self.t_full : self.x_full], label_batch_size)
    
    def label_save(self, labels) : 
        if self.t_full == self.x_full :
            msg = ("Labels have already been saved!")
            raise RuntimeError(msg)
        
        if labels.shape[0] == self.x_full - self.t_full :
            self.t = torch.cat([self.t, labels], dim=0)
            self.t_full = self.t.shape[0]
        else :
            msg = ("Number of labels=" + str(labels.shape[0]) + 
                   " does not match required number=" + str((self.x_full - 
                   self.t_full)) + "!")
            raise RuntimeError(msg)
        
    def train_load(self, train_batch_size) :
        if self.x_full != self.t_full :
           msg = ("Labels have not been saved!")
           raise RuntimeError(msg)
           
        return DataLoader(self, train_batch_size, shuffle=True)
    
    # Reservoir Sampling 시 load 함수 
    def reservoir_load(self, aug_batch_size) :
        if self.x_full != self.t_full :
           msg = ("Labels have not been saved!")
           raise RuntimeError(msg)
        
        return xtIterator(self.x[: self.k], self.t[: self.k], aug_batch_size)
    
    def aug_save(self, augs) :
        if self.x_full != self.t_full :
           msg = ("Labels have not been saved!")
           raise RuntimeError(msg)
        
        if augs.shape[0] == self.k : 
            self.x = torch.cat([self.x, augs], dim=0)
            self.x_full = self.x.shape[0]
        else :
            msg = ("Number of augs=" + str(augs.shape[0]) + 
                   " does not match required number=" + str(self.k) + "!")
            raise RuntimeError(msg)