import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Dueling QNetwork for function aproximation. 
"""

class Dueling_QNetwork(nn.Module):
    def __init__(self,action_space,state_space,seed):
        super(Dueling_QNetwork,self).__init__()
        """
        Rescale the last layer gradients by 1/sqrt(2)
        clip gradients whose norms are <= 10
        """
        self.action_space = action_space
        self.state_space = state_space
        self.seed = torch.manual_seed(seed)

        self.fc1 = nn.Linear(state_space,64)
        self.fc2 = nn.Linear(64,32)
        self.qfc1 = nn.Linear(32,action_space)
        self.vfc1 = nn.Linear(32,1)

    def forward(self,state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        q = F.relu(self.Qfc1(x))
        v = F.relu(self.Vfc1(x))
        # Max formulation
        # q_max = torch.max(q)
        # q.sub_(max)
        mean = torch.mean(q)
        q.sub_(mean)
        return torch.add(q,v)