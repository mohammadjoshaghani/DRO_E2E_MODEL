import sys
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
# torch.set_default_dtype(torch.float64)

class DataSet():
    def __init__(self):
        batch_size = 105
        LB = 32    # look-back window
        EH = 13     # evalution ahead
        FH = 1      # forecast ahead
        LG = 0      # time lagged between predictors and returns

        X = pd.read_pickle('dataset/X_iran.pkl')[:-(1+LG+EH)]     # initial size: 1135*8
        Y = pd.read_pickle('dataset/Y_iran.pkl')[1+LG:]           # initial size: 1135*20

        # self.indx = Y.index
        self.assets_name = Y.columns.tolist()

        X = torch.tensor(X.values)
        Y = torch.tensor(Y.values)


        X = X.unfold(0, LB,1).permute(0,2,1) # batch, length, features :(550, 32, 20)
        Y = Y.unfold(0, LB+EH,1).permute(0,2,1) # batch, length, features :(550, 45, 20)
        
        X = X.unfold(0, batch_size,1).permute(0,3,2,1)# 986*105*8*32
        Y = Y.unfold(0, batch_size,1).permute(0,3,2,1)# 986*105*20*45

        # split : train, valid, test
        l = np.cumsum([0.7, 0.1, 0.2])
        s = list(map(lambda x: int(x*len(Y)), l))

        # run it on GPU if availabe:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        X_train, X_valid, X_test = X[:s[0]].to(device), X[s[0]:s[1]].to(device), X[s[1]:s[2]].to(device)
        Y_train, Y_valid, Y_test = Y[:s[0]].to(device), Y[s[0]:s[1]].to(device), Y[s[1]:s[2]].to(device)

        # create train data loader
        train_dataset = TensorDataset(X_train, Y_train)
        self.train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=0)
        
        # create validation data loader
        valid_dataset = TensorDataset(X_valid, Y_valid)
        self.valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=0)

        # create test data loader
        test_dataset = TensorDataset(X_test, Y_test)
        self.test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
    
    def get(self, mode):
        if mode =='train':
            return self.train_loader
        elif mode =='valid':
            return self.valid_loader            
        elif mode =='test':
            return self.test_loader

if __name__=="__main__":
    data = DataSet()
    print(len(data.train_loader))
    print(len(data.valid_loader))
    print(len(data.test_loader))