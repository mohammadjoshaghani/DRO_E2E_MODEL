import torch

## used in DEEP-KERNEL / simple mlp-prediction
class MlpLayer(torch.nn.Sequential):
    """this takes input of size: 215*8 -> 215*out_dim
    """
    def __init__(self, input_dim=8, out_dim=20):
        super(MlpLayer, self).__init__()
        midle_dim = int((out_dim+input_dim)/2)
        self.add_module('linear1', torch.nn.Linear(input_dim, midle_dim))         
        self.add_module('relu1', torch.nn.ReLU())
        self.add_module('dropout1', torch.nn.Dropout(p=0.3))
        self.add_module('linear2', torch.nn.Linear(midle_dim, out_dim))         
        self.add_module('relu2', torch.nn.ReLU())
        self.add_module('dropout2', torch.nn.Dropout(p=0.3))
        self.add_module('linear3', torch.nn.Linear(out_dim, out_dim))