import sys
sys.path.append(r'../')

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import scipy.linalg as la
import time

from lib.models import PIVAE_SDE, MMD_loss
from lib.data_loader import trainingset_construct_SDE
from lib.visualization import *

# convey the parameters from command line
import argparse
parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--case', type=str, default = None)
parser.add_argument('--data_size', type=int, default = 2000)
parser.add_argument('--u_sensor', type=int, default = None)
parser.add_argument('--k_sensor', type=int, default = None)
parser.add_argument('--f_sensor', type=int, default = None)
parser.add_argument('--latent_dim', type=int, default = 4)
parser.add_argument('--batch_val', type=int, default = 1000)
parser.add_argument('--epoch', type=int, default = 100)
parser.add_argument('--lr', type=float, default = 0.001)
parser.add_argument('--mesh_size', type=int, default = 400)
args = parser.parse_args()

if torch.cuda.is_available():
    device_name = 'cuda'
else:
    device_name = 'cpu'
device = torch.device(device_name)

# define loss function
def criterion(u, k, f, u_recon, k_recon, f_recon, z, device):
    MMD = MMD_loss()
    recon = torch.cat((u_recon, k_recon, f_recon), dim = 1)
    ref = torch.cat((u, k, f), dim = 1)
    loss_recon = MMD(recon,ref) 
    zc = torch.randn_like(z).to(device)
    KLD = MMD(z, zc)
    
    return KLD + loss_recon 

# training function
def train(epoch,train_loader,model,optimize_operator,criterion,device):
    train_loss = 0
    for batch_idx, (u, k, f, u_coor, k_coor, f_coor) in enumerate(train_loader):
        u = u.to(device)
        k = k.to(device)
        f = f.to(device)
        u_coor = u_coor.to(device)
        k_coor = k_coor.to(device)
        f_coor = f_coor.to(device)

        optimize_operator.zero_grad()
        u_recon, k_recon, f_recon, Z = model.forward(u, k, f, u_coor, k_coor, f_coor)
        loss = criterion(u, k, f, u_recon, k_recon, f_recon, Z, device)
        loss.backward()
        train_loss += loss.item()
        optimize_operator.step()
    
    return model, loss
    
# load the data from database
u_data = np.load(file=r'../database/SDE/u_{}.npy'.format(args.case))[0:args.data_size]
k_data = np.load(file=r'../database/SDE/k_{}.npy'.format(args.case))[0:args.data_size]
f_data = np.load(file=r'../database/SDE/f_{}.npy'.format(args.case))[0:args.data_size]


# calculate ground true for comparison
if args.case == args.case:
    n_validate = 201    # number of validation points
    test_coor = np.floor(np.linspace(0,1,n_validate) * args.mesh_size).astype(int)
    u_test = u_data[:,test_coor]
    k_test = k_data[:,test_coor]
    f_test = f_data[:,test_coor]
    true_mean_u = torch.mean(torch.from_numpy(u_test),axis=0).type(torch.float).to(device)
    true_std_u = std_cal(torch.from_numpy(u_test)).type(torch.float).to(device)
    true_mean_k = torch.mean(torch.from_numpy(k_test),axis=0).type(torch.float).to(device)
    true_std_k = std_cal(torch.from_numpy(k_test)).type(torch.float).to(device)
    # calculate u reference solution
    std = np.std(u_data, axis=0)
    mean = np.mean(u_data, axis=0)
    low = mean - std
    up = mean + std
    u_ref = np.vstack((low,mean,up))
    # calculate ukreference solution
    std = np.std(k_data, axis=0)
    mean = np.mean(k_data, axis=0)
    low = mean - std
    up = mean + std
    k_ref = np.vstack((low,mean,up))

# define models
nblock = 3   # 3 blocks = 4 hidden layers
width = 128
model = PIVAE_SDE(args.latent_dim, args.u_sensor, args.k_sensor, args.f_sensor, nblock, width, device).to(device) 
optimize_operator = optim.Adam(model.parameters(), lr=args.lr, betas=(0.5,0.9))

# define training data loader
u_coor = np.linspace(-1,1,args.u_sensor) * np.ones([len(u_data),args.u_sensor])
k_coor = np.linspace(-1,1,args.k_sensor) * np.ones([len(k_data),args.k_sensor])
if args.k_sensor == 1:
    k_coor = - np.ones([len(k_data), args.k_sensor])
f_coor = np.linspace(-1,1,args.f_sensor) * np.ones([len(f_data),args.f_sensor])
x_u_coor = np.floor(np.linspace(0,1,args.u_sensor) * args.mesh_size).astype(int)
x_k_coor = np.floor(np.linspace(0,1,args.k_sensor) * args.mesh_size).astype(int)
if args.k_sensor == 1:
    x_k_coor = [0]
x_f_coor = np.floor(np.linspace(0,1,args.f_sensor) * args.mesh_size).astype(int)
k_training_data = k_data[0:args.data_size, x_k_coor]
u_training_data = u_data[0:args.data_size, x_u_coor]
f_training_data = f_data[0:args.data_size, x_f_coor]   
VAE_train_loader = trainingset_construct_SDE(u_data=u_training_data, k_data=k_training_data, f_data=f_training_data, 
                                         x_u=u_coor, x_k=k_coor, x_f=f_coor, batch_val=args.batch_val)

# train the network
u_mean_error = []
u_std_error = []
k_mean_error = []
k_std_error = []
time_history = []
loss_history = []
if __name__ == "__main__":
    for epoch in range(args.epoch):

        if epoch % 100 == 0:
            print('epoch:', epoch)

            with torch.no_grad():
                z = torch.randn(1000, args.latent_dim).to(device)
                coordinate = (torch.linspace(-1,1,steps=n_validate) * torch.ones((1000,n_validate))).to(device)
                u_recon = model.u_decoder(model.combine_xz(coordinate, z)).view(-1,n_validate)
                k_recon = model.k_decoder(model.combine_xz(coordinate, z)).view(-1,n_validate)
                
                mean_u = torch.mean(u_recon,axis=0)
                std_u = std_cal(u_recon)
                mean_k = torch.mean(k_recon,axis=0)
                std_k = std_cal(k_recon)
                
                # mean_error_forward.append(torch.norm(mean-true_mean))
                # std_error_forward.append(torch.norm(std-true_std))
                mean_L2_error_u = (torch.norm(mean_u-true_mean_u)/torch.norm(true_mean_u)).cpu().numpy()
                std_L2_error_u = (torch.norm(std_u-true_std_u)/torch.norm(true_std_u)).cpu().numpy()
                u_mean_error.append(mean_L2_error_u)
                u_std_error.append(std_L2_error_u)
                print('u mean error:', mean_L2_error_u, 'u std error:', std_L2_error_u)
                
                mean_L2_error_k = (torch.norm(mean_k-true_mean_k)/torch.norm(true_mean_k)).cpu().numpy()
                std_L2_error_k = (torch.norm(std_k-true_std_k)/torch.norm(true_std_k)).cpu().numpy()
                k_mean_error.append(mean_L2_error_k)
                k_std_error.append(std_L2_error_k)
                print('k mean error:', mean_L2_error_k, 'k std error:', std_L2_error_k)

        time_start = time.time()
        model, L = train(epoch, VAE_train_loader, model, optimize_operator, criterion, device_name)
        time_stop = time.time()
        loss_history.append(L.detach().cpu().numpy())
        time_history.append(time_stop-time_start)
    
    # torch.save(model, './results/special_case/f_high/model_fhigh.pkl')
    torch.save(model, r'./trained_model/SDE/model_{}_u={}_k={}_f={}_datasize={}_z={}.pkl'.format(args.case, args.u_sensor, args.k_sensor, args.f_sensor, args.data_size, args.latent_dim))























