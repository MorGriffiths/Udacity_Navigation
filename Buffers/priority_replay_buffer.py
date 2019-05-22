import numpy as np
import random
import torch
from collections import namedtuple, deque

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

"""
Priority Buffer HyperParameters
alpha(priority or w) dictates how biased the sampling should be towards the TD error. 0 < a < 1
beta(IS) informs the importance of the sample update

paper uses a sum tree to calculate the priority sum in O(log n) time
"""

class PriorityReplayBuffer(object):
    def __init__(self,action_size,buffer_size,batch_size,seed,priority):
        
        self.seed = random.seed(seed)
        self.action_size = action_size
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.priority = priority

        self.experience = namedtuple('experience',field_names=['state','action','reward','next_state','done','priority'])
        self.memory = deque(maxlen=buffer_size)
    
    def sample(self):
        # Super inefficient to sum everytime. We could implement the tree sum structure. 
        # Or we could sum once on the first sample and then keep track of what we add and lose from the buffer.
        TD_sum = self.TD_sum()
        # priority^a over the sum of the priorities^a = likelyhood of the given choice
        probability_distribution = [(e[5]**self.priority/TD_sum) for e in self.memory]
        experiences = np.random.choice(self.memory,self.batch_size,p=probability_distribution)
        
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None])).uint8().to(device)
        priorities = torch.from_numpy(np.vstack([e.priority for e in experiences if e is not None])).float().to(device)
        
        return (states,actions,rewards,next_states,dones,priorities)
    
    def TD_sum(self):
        TD_error_sum = sum([e[5]**self.priority for e in self.memory])
        return TD_error_sum
    
    def add(self,state,action,reward,next_state,done,priority):
        e = self.experience(state,action,reward,next_state,done,priority)
        self.memory.append(e) 
    
    def __len__(self):
        return len(self.memory)