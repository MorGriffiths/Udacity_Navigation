
import torch.optim as optim
import random
import torch
import numpy as np

from Networks/dueling_qnetwork.py import Dueling_QNetwork
from Buffers/priority_replay_buffer.py import PriorityReplayBuffer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
"""
DQN with Priority Replay, DDQN, and Dueling DQN.
"""
class Priority_DQN(object):
    def __init__(self,state_space,action_space,seed,update_every,batch_size,buffer_size,min_buffer_size,learning_rate,GAMMA,tau,clip_norm,alpha):
        self.action_space = action_space
        self.state_space = state_space
        self.seed = random.seed(seed)
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.min_buffer_size = min_buffer_size
        self.learning_rate = learning_rate
        self.update_every = update_every
        self.GAMMA = GAMMA
        self.alpha = alpha
        self.tau = tau
        self.clip_norm = clip_norm
        
        self.qnetwork_local = Dueling_QNetwork(state_space,action_space,seed)
        self.qnetwork_target = Dueling_QNetwork(state_space,action_space,seed)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(),lr=learning_rate)
        # Initialize replaybuffer
        self.memory = PriorityReplayBuffer(action_space,buffer_size,batch_size,seed,alpha)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        
    def step(self,state,action,reward,next_state,done,index):
        # Calculate TD error
        # Target
        current_network_action = self.qnetwork_local(next_state).max(1)[1]
        # initial state comes in as (1,4), squeeze to get (4)
        target = reward + self.GAMMA*(self.qnetwork_target(next_state).squeeze(0)[current_network_action])
        # Local. same rational for squeezing
        local = self.qnetwork_local(state).squeeze(0)[action]
        TD_error = reward + target - local
        # Save the experience
        self.memory.add(state,action,reward,next_state,done,TD_error,index)
        
        # learn from the experience
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            if len(self.memory) > self.min_buffer_size:
                experiences,indicies,weights = self.memory.sample(index)
                self.learn(experiences,indicies,weights)
        
    def act(self,state,eps=0.):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()
        
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return np.random.choice(np.arange(self.action_space))
        
    def learn(self,experiences,indicies,weights):
        
        states,actions,rewards,next_states,dones = experiences
        # Local max action
        local_next_state_actions = self.qnetwork_local(next_states).max(1)[1].unsqueeze(1)
        # Target
        target_values = self.qnetwork_target(next_states).detach()
        max_target = target_values.gather(1,local_next_state_actions)
#         print('max_target size',max_target.size())
        max_target *= (1-dones) 
        targets = rewards + (self.GAMMA*max_target)
#         print('targets',targets.size())
#         targets = rewards + self.GAMMA*(target_values.gather(1,local_next_state_actions))
        # Local
        local = self.qnetwork_local(states).gather(1,actions)
        TD_error = local - targets
        loss = ((torch.tensor(weights) * TD_error)**2*0.5).mean()
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.qnetwork_local.parameters(),self.clip_norm)
        self.optimizer.step()
        # Update the priorities
        TD_errors = np.abs(TD_error.squeeze(1).detach().cpu().numpy())
        self.memory.sum_tree.update_priorities(TD_errors,indicies)
        self.update_target()

    # def update_target(self,tau):
    #     for local_param,target_param in zip(self.qnetwork_local.parameters(),self.qnetwork_target.parameters()):
    #         target_param.data.copy_(local_param.data)
        
    # Polyak averaging  
    def update_target(self):
        for local_param,target_param in zip(self.qnetwork_local.parameters(),self.qnetwork_target.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1-self.tau)*target_param.data)
#         self.qnetwork_local.parameters() = TAU*self.qnetwork_local.parameters() + (1-TAU)*self.qnetwork_target.parameters()