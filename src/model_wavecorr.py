import numpy as np
import cvxpy as cp
import torch
from wavecorr import WaveCorr
from decisionLayer import DecisionLayer

class Model_WaveCorr(torch.nn.Module):
    def __init__(self, train_gamma, train_delta, data_loader, distance='HL', 
                num_latents = 10, num_tasks = 20, in_dim=8, out_dim=2, n_idc_points=90):
        
        super(Model_WaveCorr, self).__init__()
        # creating wavecorr predictor 
        self.predLayer = WaveCorr()         
        
        # Decision Layer initialization
        self.decLayer = DecisionLayer(distance).Declayer
        self.n_obs=104
        self.train_gamma = train_gamma
        self.train_delta = train_delta
        self._init_param()

    def forward(self, x, y):
        # prediction Layer:
        y_var, y_pred, gploss = self.predLayer(x,y[:self.n_obs+1,:])   # (105*20)
        y_hat =  y_pred[-1:]
        eps = y_var[:-1]         
        
        # decision Layer:
        z_star, = self.decLayer(eps, y_hat, self.gamma, self.delta)   # (1*20*1)
        return z_star.squeeze(2), y_hat, gploss      

    def _init_param(self):
        # Register 'gamma' (risk-return trade-off parameter)
        self.gamma = torch.nn.Parameter(torch.DoubleTensor(1).uniform_(0.02, 0.1))
        self.gamma.requires_grad = self.train_gamma

        # Register 'delta' (ambiguity sizing parameter) for DR layer
        ub = (1 - 1/(self.n_obs**0.5)) / 2
        lb = (1 - 1/(self.n_obs**0.5)) / 10
        self.delta = torch.nn.Parameter(torch.DoubleTensor(1).uniform_(lb, ub))
        self.delta.requires_grad = self.train_delta    


# model = Model_WaveCorr()
# for name, param in enumerate(model.named_parameters()): print(name, '->', param)

           