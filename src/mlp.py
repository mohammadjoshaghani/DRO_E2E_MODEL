import torch


## used in DEEP-KERNEL / simple mlp-prediction
class MlpLayer(torch.nn.Sequential):
    """this takes input of size: 215*8 -> 215*out_dim"""

    def __init__(self, input_dim=8, out_dim=20):
        super(MlpLayer, self).__init__()
        # midle_dim = int((out_dim+input_dim)/2)
        self.add_module("linear1", torch.nn.Linear(input_dim, out_dim))
