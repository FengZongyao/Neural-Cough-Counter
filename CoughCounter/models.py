from torch import nn
import torch
import torchaudio
import torch.nn.functional as F
import math
import warnings


########################################################################################

class HUBERTC(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.bundle = torchaudio.pipelines.HUBERT_BASE
        self.feature_extractor = self.bundle.get_model()
        self.feature_extractor.train()
        # 10s 499 5s 249
        self.dense = nn.Linear(768,2)
    
    def forward(self,x):
        x = x.view(x.shape[0],-1)
        x,_ = self.feature_extractor(x) # (n,249,768)
        x = torch.mean(x,1)   # (n,768)
        x = self.dense(x)
        return x
    
########################################################################################

class HUBERTF(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.bundle = torchaudio.pipelines.HUBERT_BASE
        self.feature_extractor = self.bundle.get_model()
        self.feature_extractor.train()
        # 10s 499 5s 249
        self.dense = nn.Linear(768,2)
    def forward(self,x):
        x = x.view(x.shape[0],-1)
        x,_ = self.feature_extractor.extract_features(x) 
        x = x[-1]  # (n,249,768)
        x = self.dense(x) #(n,249,2)
        x = x.transpose(1,2) #(n,2,249)
        return x

########################################################################################

class HUBERT_RF(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.bundle = torchaudio.pipelines.HUBERT_BASE
        self.feature_extractor = self.bundle.get_model()
        self.feature_extractor.train()
        self.dense = nn.Linear(768,2)
        self.reg_dense = nn.Linear(768,1)

    def forward(self,x):
        x = x.view(x.shape[0],-1)
        x,_ = self.feature_extractor.extract_features(x) 
        x = x[-1]  # (n,249,768)
        x_c = self.reg_dense(x) #(n,249)
        x = self.dense(x) #(n,249,2)
        x = x.transpose(1,2) #(n,2,249)
        return x,x_c.view(-1,x.shape[-1])
    

