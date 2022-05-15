__all__  = ['MMD_loss', 'SinkhornDistance', 'VAE', 'Generator', 'Discriminator', 'PIVAE_SDE',\
             'PIGAN_Generator', 'PIGAN_Discriminator', 'PIVAE_SDE_multigroup', 'PIVAE_SPDE']

import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable

class MMD_loss(nn.Module):
    'description'
    # function class which calculates the MMD distance of 2 distributions

    def __init__(self, kernel_mul = 2.0, kernel_num = 5):
        super(MMD_loss, self).__init__()

        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        return
    
    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0])+int(target.size()[0])
        total = torch.cat([source, target], dim=0)

        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2) 
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def forward(self, source, target):
        batch_size = int(source.size()[0])
        kernels = self.guassian_kernel(source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        loss = torch.mean(XX + YY - XY -YX)
        return loss

class SinkhornDistance(nn.Module):
    r"""
    Given two empirical measures each with :math:`P_1` locations
    :math:`x\in\mathbb{R}^{D_1}` and :math:`P_2` locations :math:`y\in\mathbb{R}^{D_2}`,
    outputs an approximation of the regularized OT cost for point clouds.
    Args:
        eps (float): regularization coefficient
        max_iter (int): maximum number of Sinkhorn iterations
        reduction (string, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of
            elements in the output, 'sum': the output will be summed. Default: 'none'
    Shape:
        - Input: :math:`(N, P_1, D_1)`, :math:`(N, P_2, D_2)`
        - Output: :math:`(N)` or :math:`()`, depending on `reduction`
    """
    def __init__(self, eps, max_iter, device, reduction='none'):
        super(SinkhornDistance, self).__init__()
        
        self.device = device
        self.eps = eps
        self.max_iter = max_iter
        self.reduction = reduction
        
    def forward(self, x, y):
        # The Sinkhorn algorithm takes as input three variables :
        C = self._cost_matrix(x, y)  # Wasserstein cost function
        x_points = x.shape[-2]
        y_points = y.shape[-2]
        if x.dim() == 2:
            batch_size = 1
        else:
            batch_size = x.shape[0]

        # both marginals are fixed with equal weights
        mu = torch.empty(batch_size, x_points, dtype=torch.float,
                         requires_grad=False).fill_(1.0 / x_points).squeeze().to(self.device)
        nu = torch.empty(batch_size, y_points, dtype=torch.float,
                         requires_grad=False).fill_(1.0 / y_points).squeeze().to(self.device)

        u = torch.zeros_like(mu).to(self.device)
        v = torch.zeros_like(nu).to(self.device)
        # To check if algorithm terminates because of threshold
        # or max iterations reached
        actual_nits = 0
        # Stopping criterion
        thresh = 1e-1

        # Sinkhorn iterations
        for i in range(self.max_iter):
            u1 = u  # useful to check the update
            u = self.eps * (torch.log(mu+1e-8) - torch.logsumexp(self.M(C, u, v), dim=-1)) + u
            v = self.eps * (torch.log(nu+1e-8) - torch.logsumexp(self.M(C, u, v).transpose(-2, -1), dim=-1)) + v
            err = (u - u1).abs().sum(-1).mean()

            actual_nits += 1
            if err.item() < thresh:
                break

        U, V = u, v
        # Transport plan pi = diag(a)*K*diag(b)
        pi = torch.exp(self.M(C, U, V))
        # Sinkhorn distance
        cost = torch.sum(pi * C, dim=(-2, -1))

    # if self.reduction == 'mean':
    # cost = cost.mean()
    #         elif self.reduction == 'sum':
    #             cost = cost.sum()

        return cost

    def M(self, C, u, v):
        "Modified cost for logarithmic updates"
        "$M_{ij} = (-c_{ij} + u_i + v_j) / \epsilon$"
        return (-C + u.unsqueeze(-1) + v.unsqueeze(-2)) / self.eps

    @staticmethod
    def _cost_matrix(x, y, p=2):
        "Returns the matrix of $|x_i-y_j|^p$."
        x_col = x.unsqueeze(-2)
        y_lin = y.unsqueeze(-3)
        C = torch.sum((torch.abs(x_col - y_lin)) ** p, -1)
        return C

    @staticmethod
    def ave(u, u1, tau):
        "Barycenter subroutine, used by kinetic acceleration through extrapolation."
        return tau * u + (1 - tau) * u1

class plainBlock(nn.Module):
    'description'
    # blocks of plain neural network

    def __init__(self, width):
        super(plainBlock, self).__init__()
        self.fc1 = nn.Linear(width,width)

    def forward(self, x):
        out = self.fc1(x)
        out = torch.tanh(out)
        return out

class plainBlock_res(nn.Module):
    'description'
    # blocks blocks of NN with shortcuts(ResNN) 

    def __init__(self, width):
        super(plainBlock, self).__init__()
        self.fc1 = nn.Linear(width,width)

    def forward(self, x):
        out = self.fc1(x)
        out = torch.tanh(out)
        return out + x


# neteork of PIVAE (1D case)
class PIVAE_SDE(nn.Module):
    def __init__(self, latent_dim, u_data_dim, k_data_dim, f_data_dim, n_blocks, width, device):
        super(PIVAE_SDE, self).__init__()
        
        self.device = device

        self.u_data_dim = u_data_dim
        self.k_data_dim = k_data_dim
        self.f_data_dim = f_data_dim
        self.latent_dim = latent_dim
        
        self.encoder_blocks = nn.Sequential( *(n_blocks * [plainBlock(width)])) 
        self.encoder = nn.Sequential(nn.Linear(u_data_dim + k_data_dim + f_data_dim, width), self.encoder_blocks)
        self.encoder_mu = nn.Sequential( nn.Linear(width, latent_dim),  nn.Tanh())
        self.encoder_var = nn.Sequential( nn.Linear(width, latent_dim),  nn.Sigmoid())  
        
        self.u_decoder_blocks = nn.Sequential( *(n_blocks * [plainBlock(width)])) 
        self.u_decoder = nn.Sequential(nn.Linear(latent_dim+1, width),  nn.Tanh(), self.u_decoder_blocks, nn.Linear(width, 1))
    
        self.k_decoder_blocks = nn.Sequential( *(n_blocks * [plainBlock(width)])) 
        self.k_decoder = nn.Sequential(nn.Linear(latent_dim+1, width),  nn.Tanh(), self.k_decoder_blocks, nn.Linear(width,1))

    def reparameterize(self, mu, logvar):
        std = torch.sqrt(torch.exp(logvar))
        eps = torch.randn_like(std).to(self.device)
        return mu + std * eps

    def combine_xz(self, x, z):    # x是一个n行矩阵，z是一个列向量
        x_new = x.view(-1,1)
        z_new = torch.repeat_interleave(z, x.size(-1), dim=0) # .view(-1,self.latent_dim)
        return torch.cat((x_new,z_new),1)
    
    def encode(self, u, k, f):
        com = torch.cat((u,k,f),dim=1)
        out = self.encoder(com)
        mu = self.encoder_mu(out)
        logvar = self.encoder_var(out)
        Z = self.reparameterize(mu, logvar)
    
        return Z

    def funval_cal(self, z, u_coor, k_coor):
        u_recon = self.u_decoder(self.combine_xz(u_coor, z)).view(-1,u_coor.size(1))
        k_recon = self.k_decoder(self.combine_xz(k_coor, z)).view(-1,k_coor.size(1))

        return u_recon, k_recon
    
    def PDE_check(self, z, f_coor, device):
        x = Variable(f_coor.view(-1,1).type(torch.FloatTensor), requires_grad=True).to(device)
        z_uk = torch.repeat_interleave(z, f_coor.size(1), dim=0)
        
        val_u = torch.cat((x,z_uk),1)
        val_k = torch.cat((x,z_uk),1)

        u_PDE = self.u_decoder(val_u)
        u_PDE_x = torch.autograd.grad(outputs=u_PDE, inputs=x, grad_outputs=torch.ones(u_PDE.size()).to(device),create_graph=True, only_inputs=True)[0]
        u_PDE_xx = torch.autograd.grad(outputs=u_PDE_x, inputs=x, grad_outputs=torch.ones(u_PDE_x.size()).to(device),create_graph=True, only_inputs=True)[0]
        k_PDE = self.k_decoder(val_k)
        k_PDE_x = torch.autograd.grad(outputs=k_PDE, inputs=x, grad_outputs=torch.ones(k_PDE.size()).to(device),create_graph=True, only_inputs=True)[0]
        f_recon = -0.1*(k_PDE_x * u_PDE_x + k_PDE * u_PDE_xx).view(-1,f_coor.size(1))

        return f_recon
    
    def forward(self, u, k, f, u_coor, k_coor, f_coor):
        Z = self.encode(u, k, f)
        u_recon, k_recon = self.funval_cal(Z, u_coor, k_coor)
        f_recon = self.PDE_check(Z, f_coor, self.device)
        
        return u_recon, k_recon, f_recon, Z

# network of PIGANs
class PIGAN_Generator(nn.Module):
    def __init__(self, lat_dim, udata_dim, kdata_dim, fdata_dim, width, n_blocks):
        super(PIGAN_Generator, self).__init__()
        
        self.u_blocks = nn.Sequential( *(n_blocks * [plainBlock(width)])) 
        self.u_gen = nn.Sequential(nn.Linear(lat_dim + 1, width), self.u_blocks, nn.Linear(width, 1))
        
        self.k_blocks = nn.Sequential( *(n_blocks * [plainBlock(width)])) 
        self.k_gen = nn.Sequential(nn.Linear(lat_dim + 1, width), self.k_blocks, nn.Linear(width, 1))

        self.udata_dim = udata_dim
        self.kdata_dim = kdata_dim
        self.fdata_dim = fdata_dim
        self.lat_dim = lat_dim
        
    def combine_xz(self, x, z):
        x_new = x.view(-1,1)
        z_new = torch.repeat_interleave(z, x.size(1), dim=0) # .view(-1,self.latent_dim)
        return torch.cat((x_new,z_new),1)
    
    def reconstruct(self, z, ucoor, kcoor):
        x_u = self.combine_xz(ucoor, z)
        urecon = self.u_gen(x_u)
        x_k = self.combine_xz(kcoor, z)
        krecon = self.k_gen(x_k)
        return urecon, krecon
    
    def f_recontruct(self, z, fcoor, device):
        x = Variable(fcoor.view(-1,1).type(torch.FloatTensor), requires_grad=True).to(device)
        z_uk = torch.repeat_interleave(z, fcoor.size(1), dim=0)

        x_PDE = torch.cat((x,z_uk),1)
        u_PDE = self.u_gen(x_PDE)
        k_PDE = self.k_gen(x_PDE)
        
        # calculate derivative
        u_PDE_x = torch.autograd.grad(outputs=u_PDE, inputs=x, grad_outputs=torch.ones(u_PDE.size()).to(device),create_graph=True, only_inputs=True)[0]
        u_PDE_xx = torch.autograd.grad(outputs=u_PDE_x, inputs=x, grad_outputs=torch.ones(u_PDE_x.size()).to(device),create_graph=True, only_inputs=True)[0]
        k_PDE_x = torch.autograd.grad(outputs=k_PDE, inputs=x, grad_outputs=torch.ones(k_PDE.size()).to(device),create_graph=True, only_inputs=True)[0]
        f = -0.1*(k_PDE_x * u_PDE_x + k_PDE * u_PDE_xx)
        return f
    
    def forward(self, z, ucoor, kcoor, fcoor, device):
        
        urecon, krecon = self.reconstruct(z, ucoor, kcoor)
        f = self.f_recontruct(z, fcoor, device)

        return urecon, krecon, f
    
class PIGAN_Discriminator(nn.Module):
    def __init__(self, in_dim, width):
        super(PIGAN_Discriminator, self).__init__()
        
        self.fc1 = nn.Linear(in_dim, width)
        self.fc2 = nn.Linear(width, width)
        self.fc3 = nn.Linear(width, width)
        self.fc4 = nn.Linear(width, width)
        self.fc5 = nn.Linear(width, 1)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = torch.tanh(self.fc4(x))
        x = self.fc5(x)
        return x





