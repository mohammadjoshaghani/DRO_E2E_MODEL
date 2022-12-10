import torch
import numpy as np

class MlpLayer(torch.nn.Sequential):
    def __init__(self, input_dim=8, out_dim=20):
        super(MlpLayer, self).__init__()
        midle_dim = int((out_dim+input_dim)/2)
        self.add_module('linear1', torch.nn.Linear(input_dim, midle_dim))         
        self.add_module('relu1', torch.nn.ReLU())
        self.add_module('linear2', torch.nn.Linear(midle_dim, out_dim))         
        self.add_module('relu2', torch.nn.ReLU())
        self.add_module('linear3', torch.nn.Linear(out_dim, out_dim))         


# mlp = MlpLayer()
# x = torch.randn(105,8)
# print(mlp(x).shape)
# print(list(mlp.parameters()))
