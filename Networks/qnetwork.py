import torch
import torch.nn as nn
import torch.nn.functional as F

"""
QNetwork for function aproximation. 
"""

class QNetwork(nn.Module):
    def __init__(self,action_space,state_space,seed):
        super(QNetwork,self).__init__()
        self.action_space = action_space
        self.state_space = state_space
        self.seed = torch.manual_seed(seed)

        self.fc1 = nn.Linear(state_space,64)
        self.fc2 = nn.Linear(64,32)
        self.fc3 = nn.Linear(32,action_space)

    def forward(self,state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
