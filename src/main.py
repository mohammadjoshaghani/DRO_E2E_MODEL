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
from model import Model

class Runner():
    def __init__(self, mode):
        self.mode = mode
        self.train_gamma = True
        self.train_delta = True
        self.n_obs=104   # look-back window
        self.EH = 13     # evalution ahead
        self.FH = 1      # forecast ahead
        self.NAS = 20    # number of assets
        self.data = DataSet()
        self.model = Model(self.train_gamma, self.train_delta)
        self._init()

    def _init(self):
        self._init_hyperParam(epochs=10, lr=0.1)
        self._init_mkdir()
        self._init_checkMode()
        self.mseLoss = torch.nn.MSELoss()

    def run(self):
        self.optim = torch.optim.Adam(list(self.model.parameters()), lr=0.1)
        for epch in range(self.epochs):
            self.optim.zero_grad()
            for x, y in self.dataLoader:
                x, y = x.squeeze(0), y.squeeze(0) 
                z_star, y_hat = self.model(x, y)
                loss, mse = self.loss(z_star, y_hat, y)
                # self.optim.zero_grad()
                loss.backward()
                # self.optim.step()
                self._append(epch, z_star, loss, mse)
                self.clamp()
            self.optim.step()
            print(f'loss: {loss:.3f} \t|\t mse: {mse:.3f} \t|\t gamma: {self.model.gamma.item():.4f}')
    
    def loss(self, z, y_hat, y):
        # 0.5/20 * mse + 1/len(train) * sharpe-ratio
        mse = self.mseLoss(y_hat,y[self.n_obs:self.n_obs+self.FH,:])
        portfolio_return = y[-self.EH:]@z
        sharpe_r =  portfolio_return.mean()/portfolio_return.std()
        loss = 0.5/20 * mse + 1/len(self.dataLoader) * -sharpe_r
        return -loss, mse
    
    def _append(self, epch, z_star, loss, mse):
        self.L.append(loss.detach().numpy())
        self.MSE.append(mse.detach().numpy())
        if epch == self.epochs-1:
            self.Z.append(z_star.detach().numpy())

    def _init_forSave(self):
        self.Z      =[] # portfolio_weights
        self.L      =[] # task loss
        self.MSE    =[] # mse loss
    
    def _init_hyperParam(self,epochs=1, lr=0.1):
        self.epochs = epochs
        self.lr = lr

    def _init_dataLoader(self):
        self.dataLoader = self.data.get(self.mode)
    
    def _init_mkdir(self):
        __root_path = os.getcwd()
        path = os.path.join(__root_path,'results/')
        path += 'epochs_'+str(self.epochs) + '/'
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path
        pass    
    
    def _init_checkMode(self):
        self._init_forSave()
        self._init_dataLoader()
        # reload model for valid and test phase
        if self.mode !='train':
            self.model.load_state_dict(torch.load(self.path+'/model.pt'))

    def save(self):
        self.Z = np.array(self.Z).reshape(-1,self.NAS)
        self.L = np.array(self.L).reshape(-1,1)
        self.MSE = np.array(self.MSE).reshape(-1,1)
        np.savetxt(self.path + '/Z_'+str(self.mode)+'.csv', self.Z, delimiter=",")
        np.savetxt(self.path + '/loss_'+str(self.mode)+'.csv', self.L, delimiter=",")
        np.savetxt(self.path + '/mse_'+str(self.mode)+'.csv', self.MSE, delimiter=",")
        # np.save(self.path +'/Sigma_'+str(self.mode)+'.npy', self.S)
        if self.mode =='train':
            torch.save(self.model.state_dict(), self.path+'/model.pt')
        # w = np.genfromtxt('w.csv',delimiter=",")
            # Ensure that gamma, delta > 0 after taking a descent step
            # Ensure that gamma, delta > 0 after taking a descent step
    
    def clamp(self):
        # Ensure that gamma, delta > 0 after taking a descent step
        self.model.gamma.data.clamp_(0.0001)
        self.model.delta.data.clamp_(0.0001)


mode = 'train'
runner = Runner(mode)
runner.run()
runner.save()

print('\n #################### test phase starts: \n')
# runner.mode='test'
# runner._init_checkMode()
# runner.run()
# runner.save()
