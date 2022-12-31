import torch
import numpy as np

class Equally_Weighted(torch.nn.Module):
    def __init__(self):
        super(Equally_Weighted, self).__init__()
        self.gamma = torch.nn.Parameter(torch.ones(1), requires_grad=False)  #!
        self.delta = torch.nn.Parameter(torch.ones(1), requires_grad=False) #!
        self.NAS = 20 # number of assets
        z = np.array([1.0/self.NAS for i in range(self.NAS)])
        self.z = torch.tensor(z, requires_grad=True).unsqueeze(0) # (1*20)

    def forward(self, x, y):
        # equally_weighted:
        return self.z, torch.tensor(0.0), torch.tensor(0.0)         

if __name__=="__main__":
    model = Equally_Weighted()
    x= torch.randn(105,8,32)
    y= torch.randn(105,20,45)
    w, _, _ = model(x,y)
    print(w.shape)
    print(w)