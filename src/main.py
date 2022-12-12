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
from logger import logger

class Runner():
    def __init__(self, mode):

        self.n_obs=104   # look-back window
        self.EH = 13     # evalution ahead
        self.FH = 1      # forecast ahead
        self.NAS = 20    # number of assets
        self.data = DataSet()
        self.mseLoss = torch.nn.MSELoss()
        self._init(mode)

    def _init(self, mode, epochs=20, lr=0.0125,
                train_gamma = True, train_delta = True):

        self.train_gamma = train_gamma
        self.train_delta = train_delta
        self.model = Model(self.train_gamma, self.train_delta)
        self.mode = mode
        self._init_mkdir()
        self._init_checkMode()
        self._init_hyperParam(epochs, lr)

    def run(self):
        self.optim = torch.optim.Adam(list(self.model.parameters()), lr=self.lr)
        self.optim.zero_grad()
        for epch in range(self.epochs):
            for idx , (x, y) in enumerate(self.dataLoader):
                # get decisions and predictions
                x, y = x.squeeze(0), y.squeeze(0) 
                z_star, y_hat = self.model(x, y)
                # get loss and gradients
                loss, mse, sharpe_r = self.loss(z_star, y_hat, y)
                loss.backward()
                # in train mode, the model learns with cumulative gradients.
                # in test mode, we train model with previous data in test batch.
                # we only save model-parameters in train mode.
                if self.mode != 'train' or idx == len(self.dataLoader)-1:        
                    self.optim.step()
                    self.optim.zero_grad()
                    self._clamp()
                    self._logg(loss, mse, sharpe_r)
                self._append(epch, z_star, loss, mse)
           
    def loss(self, z, y_hat, y):
        # 0.5/20 * mse + 1/len(train) * sharpe-ratio
        mse = self.mseLoss(y_hat,y[self.n_obs:self.n_obs+self.FH,:])
        portfolio_return = y[-self.EH:]@z
        sharpe_r =  -portfolio_return.mean()/portfolio_return.std()
        loss = 0.5/20 * mse + 1/len(self.dataLoader) * sharpe_r
        return loss, mse, sharpe_r
 
    def _logg(self, loss, mse, sharpe_r):
        srt_ = f'loss: {loss:9.6f}| mse: {mse:6.3f}| Sharpe: {sharpe_r:6.3f}| gamma: {self.model.gamma.item():7.4f}'
        logger.info(srt_)
     
    def _append(self, epch, z_star, loss, mse):
        self.L.append(loss.detach().numpy())
        self.MSE.append(mse.detach().numpy())
        if epch == self.epochs-1:
            self.Z.append(z_star.detach().numpy())

    def _init_forSave(self):
        self.Z      =[] # portfolio_weights
        self.L      =[] # task loss
        self.MSE    =[] # mse loss
    
    def _init_hyperParam(self, epochs, lr):
        self.epochs = epochs
        self.lr = lr
        if self.mode!='train':
            self.epochs=1

    def _init_dataLoader(self):
        self.dataLoader = self.data.get(self.mode)
    
    def _init_mkdir(self):
        __root_path = os.getcwd()
        path = os.path.join(__root_path,'results/')
        path += str('MlpLayer')+ '/'
        path += 'epochs_'+str(self.epochs) + '/'
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path
        pass    
    
    def _init_checkMode(self,mode):
        self.mode = mode
        self._init_forSave()
        self._init_dataLoader()
        # reload model for valid and test phase
        if self.mode !='train':
            try:
                self.model.load_state_dict(torch.load(self.path+'/model.pt'))
            except:
                raise ValueError (f"there is no trained model with providing hayper-parameters.\n\
                    you have to train model first!")

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
    
    def _clamp(self):
        # Ensure that gamma, delta > 0 after taking a descent step
        self.model.gamma.data.clamp_(0.0001)
        self.model.delta.data.clamp_(0.0001)

    def portfolio(self):
        portfolio_return = self.z@self.y
        portfolio_value = (1+portfolio_return)

logger.info("start:\n")

mode ='train' 
runner = Runner(mode)
# runner.run()
# runner.save()

# logger.info("####\ntest starts:\n")
# runner._init(mode='test')
# runner.run()
# runner.save()
# logger.info("\n finish.")