
import torch.optim as optim
import random
import torch
import torch.nn.functional as F
import numpy as np

from qnetwork import QNetwork
from replay_buffer import ReplayBuffer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class DQN(object):
    def __init__(self,state_space,action_space,seed,update_every,batch_size,buffer_size,learning_rate):
        self.action_space = action_space
        self.state_space = state_space
        self.seed = random.seed(seed)
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.learning_rate = learning_rate
        self.update_every = update_every
        
        self.qnetwork_local = QNetwork(state_space,action_space)
        self.qnetwork_target = QNetwork(state_space,action_space)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(),lr=learning_rate)
        # Initialize replaybuffer
        self.memory = ReplayBuffer(action_space,buffer_size,buffer_size,seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        
    def step(self,state,action,reward,next_state,done,GAMMA):
        # Save the experience
        self.memory.add_experience(state,action,reward,next_state,done)
        
        # learn from the experience
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            if len(self.memory) > self.buffer_size:
                experiences = self.memory.sample()
                self.learn(experiences,GAMMA)
        
    def act(self,state,eps=0.):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()
        
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.sample(np.arange(self.action_space))
        
    def learn(self,experiences,GAMMA):
        
        states,actions,rewards,next_states,dones = experiences
        
        target_values = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        targets = reward + (GAMMA * target_values * (1-done))
        action_values = self.qnetwork_local(states).gather(1,actions)
        loss = F.mse_loss(action_values,targets)
        loss.backward()
        self.optimizer.step()
        soft_update(TAU)
        
    def soft_update(self,tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        """
        for local_param,target_param in zip(self.qnetwork_local.parameters(),self.qnetwork_target.parameters()):
            local_param.data.copy_(tau*local_param.data + (1-tau)*target_param.data)
#         self.qnetwork_local.parameters() = TAU*self.qnetwork_local.parameters() + (1-TAU)*self.qnetwork_target.parameters()