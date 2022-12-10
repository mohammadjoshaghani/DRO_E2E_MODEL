import torch
import numpy as np

class MlpLayer(torch.nn.Sequential):
    def __init__(self, input_dim=8, out_dim=20):
        super(MlpLayer, self).__init__()
        self.add_module('linear1', torch.nn.Linear(input_dim, out_dim))         
        self.add_module('relu3', torch.nn.ReLU())

# mlp = MlpLayer()
# x = torch.randn(105,8)
# print(mlp(x).shape)
# print(list(mlp.parameters()))
