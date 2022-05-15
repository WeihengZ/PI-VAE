__all__ = ['trainingset_construct_GP', 'trainingset_construct_SDE', 
           'trainingset_construct_SDE_multigroup', 'trainingset_construct_SPDE']

import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch

def training_loader_construct(dataset,batch_num):
    'description'
    # construct the train loader given the dataset and batch size value
    # this function can be used for all different cases 

    train_loader = DataLoader(
        dataset,
        batch_size=batch_num,
        shuffle=True,                     # change the sequence of the data every time
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    return train_loader

# define training loader construction function
class MyDataset_SDE(Dataset):
    def __init__(self, u_data, k_data, f_data, x_u, x_k, x_f, transform=None):
        self.u_data = torch.from_numpy(u_data).float()
        self.k_data = torch.from_numpy(k_data).float()
        self.f_data = torch.from_numpy(f_data).float()
        self.x_u = torch.from_numpy(x_u).float()
        self.x_k = torch.from_numpy(x_k).float()
        self.x_f = torch.from_numpy(x_f).float()
        self.transform = transform
        
    def __getitem__(self, index):
        u = self.u_data[index]
        k = self.k_data[index]
        f = self.f_data[index]
        u_coor = self.x_u[index]
        k_coor = self.x_k[index]
        f_coor = self.x_f[index]
        
        if self.transform:
            u = self.transform(u)
            k = self.transform(k)
            f = self.transform(f)
        
        return u, k, f, u_coor, k_coor, f_coor
    
    def __len__(self):
        return len(self.u_data)     # batch index 's total number 

def trainingset_construct_SDE(u_data, k_data, f_data, x_u, x_k, x_f, batch_val):
    VAE_dataset = MyDataset_SDE(u_data, k_data, f_data, x_u, x_k, x_f)
    VAE_train_loader = training_loader_construct(dataset = VAE_dataset,batch_num = batch_val)

    return VAE_train_loader







