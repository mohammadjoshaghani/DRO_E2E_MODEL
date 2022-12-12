from GP import *
import numpy as np
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
import torch

from decisionLayer import DecisionLayer

class ModelGP(torch.nn.Module):
    def __init__(self, train_gamma, train_delta, Dloader='', N=20, in_dim=8, out_dim=20, BATCH_SIZE=1):
        super(ModelGP, self).__init__()
        # Gaussian process initialization
        # init_data, _ = next(iter(Dloader))
        inducing_points = torch.randn(105,20,1)               # this should be from real input
        self.predLayer = MainGp(inducing_points, in_dim ,out_dim, BATCH_SIZE, len(Dloader.dataset))
        # Decision Layer initialization
        self.decLayer = DecisionLayer().Declayer
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
        return z_star.squeeze(0), y_hat, gploss      

    def _init_param(self):
        # Register 'gamma' (risk-return trade-off parameter)
        self.gamma = torch.nn.Parameter(torch.DoubleTensor(1).uniform_(0.02, 0.1))
        self.gamma.requires_grad = self.train_gamma

        # Register 'delta' (ambiguity sizing parameter) for DR layer
        ub = (1 - 1/(self.n_obs**0.5)) / 2
        lb = (1 - 1/(self.n_obs**0.5)) / 10
        self.delta = torch.nn.Parameter(torch.DoubleTensor(1).uniform_(lb, ub))
        self.delta.requires_grad = self.train_delta    


# model = Model()
# for name, param in enumerate(model.named_parameters()): print(name, '->', param)

           