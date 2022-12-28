import numpy as np
import cvxpy as cp
import torch
from wavecorr import WaveCorr
from decisionLayer import DecisionLayer

class Model_WaveCorr(torch.nn.Module):
    def __init__(self, train_gamma, train_delta, 
                distance='HL', mtype="predictions"):
        
        super(Model_WaveCorr, self).__init__()
        # creating wavecorr predictor 
        self.predLayer = WaveCorr(mtype)         
        
        # Decision Layer initialization
        self.decLayer = DecisionLayer(distance).Declayer
        self.n_obs=104
        self.LB = 32 # Look-back window
        self.train_gamma = train_gamma
        self.train_delta = train_delta
        self._init_param()
        self.mse = torch.nn.MSELoss()

    def forward(self, x, y):
        # prediction Layer:
        y_pred = self.predLayer(x)   # (105*20)
        y_hat =  y_pred[-1:]
        eps = y_pred[:-1] - y[:self.n_obs, :, self.LB:self.LB+1].squeeze(2)         
        mse = self.mse(y_hat, y[self.n_obs:self.n_obs+1, :, self.LB:self.LB+1].squeeze(2))

        # decision Layer:
        z_star, = self.decLayer(eps, y_hat, self.gamma, self.delta)   # (1*20*1)
        return z_star.squeeze(2), y_hat, mse      

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

           