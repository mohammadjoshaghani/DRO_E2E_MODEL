import torch
import numpy as np

class Corr_Layer(torch.nn.Module):
    def __init__(self,):
        super(Corr_Layer, self).__init__()
        self.n_assets = 20

    def forward(self,x):
        for iii in range(self.n_assets):
            xx = torch.concat((x[:,:,iii,:], x),dim=2)
            out_i = self.conv(xx)
            stacked_out.append(out_i)
        output = torch.concat((x, tensors in  stacked_out), dim=3)
        return output

        
class WaveCorr_Block(torch.nn.Module):
    def __init__(self,):
        super(WaveCorr_Block, self).__init__()
        self.corr = Corr_Layer()

    def forward(self,x):
        x_short = x
        for i in range(2):    
            x = self.dilated_conv(x)
        x = self.corr(x)
        xx = slef.conv(x)
        x = x + xx
        x = torch.nn.ReLU(x)
        return x, x_short

class WaveCorr(torch.nn.Module):
    def __init__(self,):
        super(WaveCorr, self).__init__()
        self.n_box = 5 # number of wavecorr_block
        self.box =  WaveCorr_Block()        
    
    def forward(self,x):
        for i in range(self.n_box):
            x, x_short_i = self.box(x)
            x_short_stacked.append(x_short_i) # save each block's x_short for CONV operator and summation
        
        x_short_stacked_conv = map(self.conv(), x_short_stacked) # perform conv on each x_short_i
        
        # summation
        #x = x + all 5 tensors in x_short_stacked_conv     
        x = torch.nn.ReLU(x)
        x = torch.nn.Softmax(x)
        return x # wavecorr output    

    def conv(self,x):
        # convolution operator on each tensor
        return x




X, Y = torch.randn(105,32,8,1), torch.randn(105,45,20,1)