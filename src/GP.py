import sys, os
# __root_path = os.path.dirname(os.path.abspath(os.curdir))
# sys.path.insert(0, __root_path)

import math
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
import gpytorch
import random


from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy
from gpytorch.constraints import GreaterThan



## DEEP KERNEL
class MlpLayer(torch.nn.Sequential):
    def __init__(self, input_dim=8, out_dim=20):
        super(MlpLayer, self).__init__()
        midle_dim = int((out_dim+input_dim)/2)
        self.add_module('linear1', torch.nn.Linear(input_dim, midle_dim))         
        self.add_module('relu1', torch.nn.ReLU())
        self.add_module('linear2', torch.nn.Linear(midle_dim, out_dim))         
        self.add_module('relu2', torch.nn.ReLU())
        self.add_module('linear3', torch.nn.Linear(out_dim, out_dim)) 

# Gaussian process model
class GPModel(ApproximateGP):
    def __init__(self, inducing_points,BATCH_SIZE):
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(1),batch_shape=torch.Size([105]))
        variational_strategy = VariationalStrategy(self, inducing_points, variational_distribution, learn_inducing_locations=True)
        super(GPModel, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.PeriodicKernel()#batch_shape=torch.Size([BATCH_SIZE])
        # self.covar_module.register_constraint("raw_lengthscale", GreaterThan(3.0))
        
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


# main deep kernel gaussian process model
class DKLModel(gpytorch.Module):
    def __init__(self, inducing_points, feature_extractor, BATCH_SIZE):
        super(DKLModel, self).__init__()
        self.feature_extractor = feature_extractor
        self.gp_layer = GPModel(inducing_points, BATCH_SIZE)
        
    def forward(self, x):
        x = x.permute(0,2,1)
        features = self.feature_extractor(x)     # assets * features
        features = features.permute(0,2,1)
        res = self.gp_layer(features)
        return res        

class MainGp(torch.nn.Module):
    def __init__(self, inducing_points, in_dim=7, out_dim=3, BATCH_SIZE=1, num_data = 'len(train_loader.dataset)'):
        super(MainGp,self).__init__()
        feature_extractor = MlpLayer(in_dim, out_dim)
        self.modelgp = DKLModel(inducing_points, feature_extractor, BATCH_SIZE)                                # main model
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood(batch_shape=torch.Size([105]))
        self.mll = gpytorch.mlls.VariationalELBO(self.likelihood, self.modelgp.gp_layer, num_data=num_data)

    def forward(self,x_batch,y_batch):
        # calculate loss for one batch
        f_preds = self.modelgp(x_batch.unsqueeze(2))
        loss = -self.mll(f_preds, y_batch).mean()
        # get covariance and mean for one batch
        y_preds = self.likelihood(f_preds)
        y_mean = y_preds.mean
        y_var = y_preds.variance
        # y_covar = y_preds.covariance_matrix
        return y_var, y_mean, loss



# BATCH_SIZE = 8
# N=30
# in_dim =7
# out_dim = 3
# train_x = torch.randn(BATCH_SIZE,N,out_dim)
# num_data = 101

# asset_idx = random.sample(range(N), int(N*0.75))
# inducing_points = train_x[:BATCH_SIZE, asset_idx, :out_dim]               # 3/4 of assets
# print(MainGp(inducing_points, N, in_dim ,out_dim, BATCH_SIZE,num_data=num_data))        