import sys
import os
sys.path.insert(0, os.getcwd())

import numpy as np
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
import torch
import torch.nn as nn
torch.set_default_dtype(torch.float64)

from dataset.data import DataSet
from logger import logger
import time

class Runner():
    def __init__(self, mode, epochs, model_name, distance):
        """it takes parameters, makes model, and run it
         for train/validation/test phases. 

        Args:
            mode (str): it can be: "train", "valid", "test"
            model_name (str, optional): model architecure. it can be: "MLP", "MLP_K_GP"
            distance (str, optional): distance type in decision layer. it can be "HL", "KL"
        """

        self.n_obs=104   # number of obsevation
        self.EH = 13     # evalution ahead
        self.FH = 1      # forecast ahead
        self.NAS = 20    # number of assets
        self.LB = 32     # look-back window
        self.data = DataSet()
        self.model_name = model_name
        self.distance = distance
        self._init(mode=mode, epochs=epochs)

    def _init(self, mode, epochs, lr=0.0125,
                train_gamma = True, train_delta = True):
        
        self.mode = mode        
        self.epochs = epochs
        self.train_gamma = train_gamma
        self.train_delta = train_delta
        self._init_dataLoader()
        self._init_model()
        self._init_mkdir()
        self._init_checkMode()
        self._init_hyperParam(lr)
    
    def _init_model(self):
        if self.model_name == "MLP":
            from model_mlp import Model_MLP
            self.model = Model_MLP(self.train_gamma, self.train_delta, self.distance)

        elif self.model_name == "MLP_K_GP":     
            from model_mlpKgp import Model_MLP_K_GP
            self.model = Model_MLP_K_GP(self.train_gamma, self.train_delta,
                                    self.dataLoader, self.distance)
        elif self.model_name == "WaveCorr":
            from model_wavecorr import Model_WaveCorr
            self.model = Model_WaveCorr(self.train_gamma, self.train_delta,)                           
        
        elif self.model_name == "WaveCorr_Casual":
            from model_wavecorr import Model_WaveCorr_Casual
            self.model = Model_WaveCorr_Casual()

        elif self.model_name == "Equally_weighted":
            from equally_weighted import Equally_Weighted
            self.model = Equally_Weighted()
            self.distance = ''
        
        else:
            raise ValueError(f"\n model {self.model_name} is not implemented!.\n")
    
    def run(self):
        self.optim = torch.optim.Adam(list(self.model.parameters()), lr=self.lr, weight_decay=1e-4)
        self.optim.zero_grad()
        for epch in range(self.epochs):
            for idx , (x, y) in enumerate(self.dataLoader):
                # get decisions and predictions
                x, y = x.squeeze(0), y.squeeze(0) 
                z_star, y_hat, predLoss = self.model(x, y)
                # get loss and gradients
                loss, sharpe_r = self.loss(z_star, y, predLoss)
                loss.backward()
                # in train mode, the model learns with cumulative gradients.
                # in test mode, we train model with previous data in test batch.
                # we only save model-parameters in train mode.
                if idx % 3 == 0 or self.mode != 'train' or idx == len(self.dataLoader)-1:        
                    self.optim.step()
                    self.optim.zero_grad()
                    self._clamp()
                    self._logg(loss, predLoss, sharpe_r, epch, idx)
                self._append(epch, z_star, loss, predLoss)
        # save experiment results
        _ = self.save()    
           
    def loss(self, z, y, predLoss):
        # 0.5/20 * predLoss + 1/len(train) * sharpe-ratio
        y = y[self.n_obs:self.n_obs+1, :, -self.EH:].squeeze(0)
        portfolio_return = z@y
        sharpe_r =  portfolio_return.mean()/portfolio_return.std()       #! the sharp ratio in objective is different from portfolio evaluation
        
        #* Sharpe ratio be objective
        # loss = 0.5/20 * predLoss + 1/len(self.dataLoader) * -sharpe_r  #! arbitrary: *0.01
        
        #* portfolio_value be objective 
        prtfoli_value = torch.cumprod(1 + portfolio_return, dim=1)
        prtfolio_final_value = prtfoli_value[0,-1]
        loss = 0.5/20 * predLoss + 1/len(self.dataLoader) * -prtfolio_final_value  

        return loss, sharpe_r
 
    def _logg(self, loss, predLoss, sharpe_r, epch, idx):
        srt_ = f'epochs/idx: {epch+1:3.0f}/{idx+1:4.0f}| loss: {loss:9.6f}| predLoss: {predLoss:7.3f}| Sharpe: {sharpe_r:6.3f}| gamma: {self.model.gamma.item():7.4f}'
        logger.info(srt_)
     
    def _append(self, epch, z_star, loss, predLoss):
        self.L.append(loss.detach().numpy())
        self.predLoss.append(predLoss.detach().numpy())
        if epch == self.epochs-1:
            self.Z.append(z_star.detach().numpy())

    def _init_forSave(self):
        self.Z      =[] # portfolio_weights
        self.L      =[] # task loss
        self.predLoss =[] # gaussian process loss
    
    def _init_hyperParam(self, lr):
        self.lr = lr
        if self.mode!='train':
            self.epochs=1
        if self.model_name == "Equally_weighted":
            self.epochs=1    

    def _init_dataLoader(self):
        self.dataLoader = self.data.get(self.mode)
    
    def _init_mkdir(self):
        __root_path = os.getcwd()
        path = os.path.join(__root_path,'results/')
        path += self.model_name+ '_' 
        path += self.distance+ '/' 
        path += 'epochs_'+str(self.epochs) + '/'
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path
        pass    
    
    def _init_checkMode(self):
        self._init_forSave()
        # reload model for valid and test phase
        if self.mode !='train':
            try:
                self.model.load_state_dict(torch.load(self.path+'/model.pt'))
            except:
                raise ValueError (f"there is no trained model with providing hayper-parameters.\n\
                    you have to train model first!")

    def _get_loop_result(self):
        self.Z = np.array(self.Z).reshape(-1,self.NAS)
        self.L = np.array(self.L).reshape(-1,1)
        self.predLoss = np.array(self.predLoss).reshape(-1,1)

    def _clamp(self):
        # Ensure that gamma, delta > 0 after taking a descent step
        self.model.gamma.data.clamp_(0.0001)
        self.model.delta.data.clamp_(0.0001)

    def _portfolio(self):
        y_true = self.dataLoader.dataset.tensors[1][:,self.n_obs:self.n_obs+self.FH,:,self.LB:self.LB+1].squeeze(3)
        z = torch.tensor(self.Z).unsqueeze(2)
        self.portfolio_return = torch.bmm(y_true,z).squeeze(2)
        portfolio_value = torch.cumprod(1+self.portfolio_return, dim=0)
        mean_p = portfolio_value[-1]**(1/len(portfolio_value))-1
        std_p = torch.std(self.portfolio_return)
        self.sharpe_ratio = mean_p/std_p
        logger.info(f"\n portfolio sharpe-ratio: \
            {self.sharpe_ratio.item():5.2f}")

    def save(self):
        self._get_loop_result()
        self._portfolio()
        np.savetxt(self.path + '/Z_'+str(self.mode)+'.csv', self.Z, delimiter=",")
        np.savetxt(self.path + '/loss_'+str(self.mode)+'.csv', self.L, delimiter=",")
        np.savetxt(self.path + '/predLoss_'+str(self.mode)+'.csv', self.predLoss, delimiter=",")
        np.savetxt(self.path + '/portReturns_'+str(self.mode)+'.csv', self.portfolio_return, delimiter=",")
        
        # np.save(self.path +'/Sigma_'+str(self.mode)+'.npy', self.S)
        if self.mode =='train':
            torch.save(self.model.state_dict(), self.path+'/model.pt')
        # w = np.genfromtxt('w.csv',delimiter=",")



# Todo:implement some gp that has meaningfull learnable parameters.
# Todo:training with more epochs decrease losses.
# Todo:digest meaningfull mathematics of gp.
# Todo: experiment if learn gp seperately performs better.