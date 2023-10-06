import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from torch.optim import Adam



class FeedForwardANN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        
        # self.h_neuron = h_neuron
        
        alpha = 0.01
        self.network = torch.nn.Sequential(
            nn.Linear(input_dim, 3),
            nn.LeakyReLU(alpha),
            nn.Linear(3, 3),
            nn.LeakyReLU(alpha),
            nn.Linear(3, 1),
        )
        
        torch.manual_seed(42)
        for m in self.modules():
                if isinstance(m, nn.Linear):
                    m.weight = nn.init.kaiming_uniform_(m.weight)
                    m.bias.data.fill_(0)

    def forward(self, x):
        x = x.view([x.shape[0], -1, x.shape[1]])
        return self.network(x[:, -1, :])
    
    
