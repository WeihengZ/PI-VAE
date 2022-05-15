__all__ = ['setup_seed', 'samples_plot', 'cov_square_exp', 'cov_exp', 'PCA', 'w1_dist_for_empirical',\
           'SPDE_visual', 'std_cal']

import numpy as np
import random
import scipy.linalg as la
import scipy as sp
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import ot


# function of setting random seed
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

# ----------------------------------------- GP case ----------------------------------------

# function of ploting the sample path
def samples_plot(data, flag_sensor, flag_lengthscale):
    plt.fig, ax = plt.subplots()
    x = np.linspace(-1,1,len(data[0,:]))
    sensor_position = np.linspace(-1,1,flag_sensor)
    lower_bound = np.min(np.min(np.array(data)))
    upper_bound = np.max(np.max(np.array(data)))
    for i in range(400):
        ax.plot(x,data[i,:])
        for k in range(flag_sensor):
            plt.vlines(sensor_position[k], lower_bound, upper_bound, colors = "k", linestyles = "dashed")
    # plt.savefig('results/Effect_of_sensor_number/sample_path_l={lengthscale}_sensor={num_sensor}.jpg'.format(lengthscale=flag_lengthscale, num_sensor=flag_sensor))
    plt.show()

# function for calculating the ground true covariance of square exponential kernel
def cov_square_exp(xs, ys, l, sigma=1):
    # Pairwise difference matrix.
    dx = np.expand_dims(xs, 1) - np.expand_dims(ys, 0)
    return (sigma ** 2) * np.exp(-((dx / l) ** 2) / 2)

# function for calculating the ground true covariance of exponential kernel
def cov_exp(xs, ys, l, sigma=1):
    # Pairwise difference matrix.
    dx = np.abs(np.expand_dims(xs, 1) - np.expand_dims(ys, 0))
    return (sigma ** 2) * np.exp(-(dx / l))

# function to calculate the spectral of the samples of a distribution
def PCA(data):
    mean_val = np.mean(data, axis=0)
    data_normalize = (data - mean_val)
    cov = (1/len(data))*(data_normalize.T @ data_normalize)
    a, b = la.eig(cov)
    return a

# function of checking the distance of the distribution
def w1_dist_for_empirical(data1,data2):
    data1 = torch.tensor(data1)
    data2 = torch.tensor(data2)
    n = len(data1)
    data_dist1 = torch.zeros((n,1))
    data_dist2 = torch.zeros((n,1))
    for index in range(n):
        data_dist1[index,0] = torch.norm(data1[index,:])
        data_dist2[index,0] = torch.norm(data2[index,:])
    C1 = sp.spatial.distance.cdist(data_dist1, data_dist1)       # Compute distance between each pair of the two collections of inputs
    C2 = sp.spatial.distance.cdist(data_dist2, data_dist2)

    p = ot.unif(n)                             # return a uniform histogram of length n_samples
    q = ot.unif(n)                             # return a uniform histogram of length n_samples
    gw0, log0 = ot.gromov.gromov_wasserstein(C1, C2, p, q, 'square_loss', verbose=False, log=True)
    
    return log0['gw_dist']

# function of calculate std on the validation points
def std_cal(A):
    mean = torch.mean(A,axis=0)
    A = (A - mean)
    std = 0
    for i in range(A.size(0)):
        std += torch.norm(A[i,:])**2
    std = torch.sqrt(std / A.size(0))
    return std

# ----------------------------------------- SDE case ----------------------------------------


# ----------------------------------------- SPDE case ----------------------------------------

def SPDE_visual(samples, figsize, root=None):
    'description'
    # this function is used to plot the mean value and the std of each point all over the whole domain
    'input'
    # samples: (nD) n-dimension array with the first dimension is batch size
    'output'
    # mean: (1D) mean of the samples
    # std: (1D) std of the samples

    # reshape the samples
    std = []
    new_samples = []
    for i in range(np.size(samples, axis=0)):
        point = np.reshape(samples[i],(1,-1),order='F')
        new_samples.append(point)
    samples = np.array(new_samples).squeeze(1)

    num = np.size(samples, axis=1)
    dim = int(np.sqrt(num))
    # calculate the mean 
    mean = np.mean(samples, axis=0)
    std = np.std(samples, axis=0)
    mean_plot = np.reshape(mean, (dim,dim), order='F')
    std_plot = np.reshape(std, (dim,dim), order='F')

    # plot the results
    plt.figure(figsize = figsize)
    ax1 = plt.subplot(1,2,1)
    ax2 = plt.subplot(1,2,2)
    plt.sca(ax1)
    low = np.min(mean); high = np.max(mean)
    sns.set()
    sns.heatmap(mean_plot, vmin=low, vmax=high)   
    plt.sca(ax2)
    low = np.min(std); high = np.max(std)
    sns.set()
    sns.heatmap(std_plot, vmin=low, vmax=high)  
    if root != None:
        plt.savefig(root)
    if root == None:
        plt.show()

    return mean, std