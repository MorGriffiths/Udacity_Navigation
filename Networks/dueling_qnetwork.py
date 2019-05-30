import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Dueling QNetwork for function aproximation. Splits the network prior to the end into two streams V and Q. 
V is the estimate of the value of the state. Q is the advantage of each action given the state.
Two formulations for subtracting Q from V:
V - max(Q)
This verision makes more sense theoretically as the value of V* should equal the max(Q*(s,A)). 
But in practice mean allows for better performance.
V - mean(Q)
Same as max except now they are separated by a constant. 
And not as susceptable to over optimism due to randomness of Q values.
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
        x = state
        if not isinstance(state,torch.Tensor):
            x = torch.tensor(x,device = self.device,dtype=torch.float32)
            x = x.unsqueeze(0)
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        a = F.relu(self.Qfc1(x))
        v = F.relu(self.Vfc1(x))
        # Max formulation
        # q_max = torch.max(q)
        # q.sub_(max)
        # mean = torch.mean(a)
        # q.sub_(mean)
        # torch.add(q,v)
        v = v.expand_as(a)
        q = v + a - a.mean(1,keepdim=True).expand_as(a)
        return q