import numpy as np
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
import torch

from decisionLayer import DecisionLayer
from mlp import MlpLayer


class Model_MLP(torch.nn.Module):
    def __init__(self, train_gamma, train_delta, distance='HL'):
        super(Model_MLP, self).__init__()
        # directly maps 8 features to 20 assets returns.
        self.predLayer = MlpLayer(input_dim=8, out_dim=20) # 105*8 -> 105*20
        self.decLayer = DecisionLayer(distance).Declayer
        self.n_obs = 104
        self.FH = 1
        self.train_gamma = train_gamma
        self.train_delta = train_delta
        self._init_param()
        self.mseLoss = torch.nn.MSELoss()
    
    def forward(self, x, y):
        # prediction Layer:
        y_pred = self.predLayer(x)   # (105*20)
        y_hat =  y_pred[-1:]
        y_eps =  y_pred[:-1]
        eps = y[:self.n_obs,:] - y_eps         
        mse = self.mseLoss(y_hat,y[self.n_obs:self.n_obs+self.FH,:])

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


# model = Model_MLP()
# for name, param in enumerate(model.named_parameters()): print(name, '->', param)

       