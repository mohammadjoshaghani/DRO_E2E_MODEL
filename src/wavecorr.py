import torch
import numpy as np
from mlp import MlpLayer



def ConvModule_Box(i, di=1, di_operator=True, channel=8):
    """makes convolution function.

    Args:
        i (_type_): box number
        di (int, optional): dilation. Defaults to 1.
        di_operator(boolean, optional): True: if we want dilated operator to reduce window length.
                                        False : if we want casual cnn for feature manipulation.
        in_chnl (int, optional): CNN dimension of input-channel: out_channels + 1. Defaults to 9.

    Returns:
        torch.nn.Conv2d: return CNN model
    """
    # define convolution function
    k_size = int(32/((2**i)*di)+1)
    # first box has input_channels=1
    # other boxes are consistant and equal 9
    in_chnl, out_chnl= channel+1, channel
    if i==1:
        in_chnl=1
    if not di_operator:
        out_chnl = channel+1    
    f = torch.nn.Conv2d(in_channels=in_chnl, out_channels=out_chnl,
            kernel_size=(1,k_size), padding='valid', dilation=di)
    return f


def ConvModule(i, is_f_outPut=False, di=1, channel=8):
    # define convolution function
    k_size = int(32/2**(i-1))
    # first box has input_channels=1
    # other boxes are consistant and equal 9
    in_chnl, out_chnl= channel+1, 1
    if i==1:
        in_chnl=1
    if i==5 and is_f_outPut:
        k_size=1
    f = torch.nn.Conv2d(in_channels=in_chnl, out_channels=out_chnl,
            kernel_size=(1,k_size), padding='valid', dilation=di)
    return f


class Corr_Layer(torch.nn.Module):
    def __init__(self,):
        super(Corr_Layer, self).__init__()
        self.n_features = 8
        di = 1 # dilation
        self.conv_corr = torch.nn.Conv2d(in_channels=8, out_channels=1, kernel_size=(int(8/di)+1,1), padding='valid', dilation=di)
    
    def forward(self,x): # 105*8*8*16
        stacked_out = []
        for iii in range(self.n_features):
            xx = torch.concat((x[:,:,iii,:].unsqueeze(2), x),dim=2)  #105*8*9*16
            out_i = self.conv_corr(xx) #! 105*1*1*16
            stacked_out.append(out_i) #105*1*8*16
        stacked_out = torch.cat(stacked_out, dim=2) #105*1*8*16
        output = torch.cat((x,stacked_out),dim=1) #105*9*8*16
        return output

        
class WaveCorr_Box(torch.nn.Module):
    def __init__(self,i):
        super(WaveCorr_Box, self).__init__()
        self.corrLayer = Corr_Layer()
        self.dilated_conv1 = ConvModule_Box(i)
        self.conv_box = ConvModule_Box(i, di_operator=False)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=0.5)

    def forward(self,x):
        x_short = x #105*1*8*32 or #105*9*8*(16/8/4/2/1)
        x = self.dilated_conv1(x)#105*8*8*16
        x = self.dropout(x)
        x = self.corrLayer(x)#105*9*8*16
        xx = self.conv_box(x_short)#!105*9*8*16
        x = x + xx #105*9*8*16
        x = self.relu(x)
        return x, x_short


class WaveCorr(torch.nn.Module):
    def __init__(self, mtype = "predictions"):
        """WaveCorr model based on pytorch. 

        Args:
            mtype (str, "weights" or "predictions"): determines model's out-put is portfolio weights
                                 or return predictions . Defaults to "predictions".
        """
        super(WaveCorr, self).__init__()
        self.conv_box1, self.box1 =  ConvModule(1), WaveCorr_Box(1)        
        self.conv_box2, self.box2 =  ConvModule(2), WaveCorr_Box(2)        
        self.conv_box3, self.box3 =  ConvModule(3), WaveCorr_Box(3)        
        self.conv_box4, self.box4 =  ConvModule(4), WaveCorr_Box(4)        
        self.conv_box5, self.box5 =  ConvModule(5), WaveCorr_Box(5)        
        self.conv_lastbox = ConvModule(5, is_f_outPut=True)
        self.actions_layer = MlpLayer(input_dim=8, out_dim=20)
        assert mtype in ["weights", "predictions"], "mtype can be 'weights' or 'predictions'! "   
        self.final_layer = torch.nn.Softmax(dim=2) if (mtype == "weights") else lambda x:x
    
    def forward(self,x):
        
        x = x.unsqueeze(1)
        # apply boxes on input
        x, x_short_1 = self.box1(x) # 105*9*8*16,   105*1*8*32
        x, x_short_2 = self.box2(x) # 105*9*8*8,    105*9*8*16
        x, x_short_3 = self.box3(x) # 105*9*8*4,    105*9*8*8
        x, x_short_4 = self.box4(x) # 105*9*8*2,    105*9*8*4
        x, x_short_5 = self.box5(x) # 105*9*8*1,    105*9*8*2

        # perform conv on each x_short_i
        x_short_conv = []
        for i in range(1,6):
            x_short_conv.append(eval(f'self.conv_box{i}(x_short_{i})'))
        x_short_conv.append(self.conv_lastbox(x))
        x_short_stacked = torch.stack(x_short_conv)
        del(x_short_conv)

        # summation
        x = x_short_stacked.sum(0)
        x = self.actions_layer(x.squeeze(3)) # 105*1*8*1 -> 105*1*20
        x = self.final_layer(x) # Todo: write sotmax for MLP/ MLP_K_GP too!
        
        return x.squeeze(1) # wavecorr output    


if __name__ == "__main__":

    x = torch.randn(105,1,8,32)
    model = WaveCorr()
    y = model(x)
    print(y.shape)
    print(y)
