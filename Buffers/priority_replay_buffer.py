import numpy as np
import random
import torch
from collections import namedtuple, deque
from priority_tree import PriorityTree,propogate

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

"""
Priority Buffer HyperParameters
alpha(priority or w) dictates how biased the sampling should be towards the TD error. 0 < a < 1
beta(IS) informs the importance of the sample update

The paper uses a sum tree to calculate the priority sum in O(log n) time. As such, i've implemented my own version
of the sum_tree which i call priority tree.
"""

class PriorityReplayBuffer(object):
    def __init__(self,action_size,buffer_size,batch_size,seed,alpha=0.5,beta=0.5,beta_end=1,beta_duration=1e-5,epsilon=7e-5):
        
        self.seed = random.seed(seed)
        self.action_size = action_size
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.alpha = alpha
        self.beta = beta
        self.beta_end = beta_end
        self.beta_duration = beta_duration
        self.beta_increment = (beta - beta_end) / beta_duration
        self.max_w = 0
        self.epsilon = epsilon
        self.TD_sum = 0

        self.experience = namedtuple('experience',field_names=['state','action','reward','next_state','done','i_episode'])
        self.sum_tree = PriorityTree(buffer_size,batch_size,alpha,epsilon)
        self.memory = {}
    
    def add(self,state,action,reward,next_state,done,TD_error,i_episode):
        e = self.experience(state,action,reward,next_state,done,i_episode)
        index = i_episode % self.buffer_size
        # add memory to memory and add corresponding priority to the priority tree
        self.memory[index] = e
        sum_tree.add(TD_error,self.index)

    def sample(self):
        # We times the error by these weights for the updates
        # Super inefficient to sum everytime. We could implement the tree sum structure. 
        # Or we could sum once on the first sample and then keep track of what we add and lose from the buffer.
        # priority^a over the sum of the priorities^a = likelyhood of the given choice
        # Anneal beta
        self.update_beta()
        # Get the samples and indicies
        priorities,indicies = self.sum_tree.sample()
        # Normalize with the sum
        norm_priorities = priorities / self.sum_tree.root.value
        samples = self.memory[indicies]
        # Importance weights
        importances = [(priority * self.buffer_size)**-self.beta for priority in norm_priorities]
        self.max_w = max(self.max_w,max(importances))
        # Normalize importance weights
        norm_importances = [importance / self.max_w for importance in importances]
        
        states = torch.from_numpy(np.vstack([e.state for e in samples if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in samples if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in samples if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in samples if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in samples if e is not None])).uint8().to(device)
        
        return (states,actions,rewards,next_states,dones,indicies,norm_importances)

    def update_beta(self):
        self.beta += self.beta_increment
        self.beta = min(self.beta,self.beta_end)
    
    def __len__(self):
        return len(self.memory)