
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
    def __init__(self,state_space,action_space,seed,update_every,batch_size,buffer_size,learning_rate,GAMMA,tau,clip_norm):
        self.action_space = action_space
        self.state_space = state_space
        self.seed = random.seed(seed)
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.learning_rate = learning_rate
        self.update_every = update_every
        self.GAMMA = GAMMA
        self.tau = tau
        self.clip_norm = clip_norm
        
        self.qnetwork_local = Dueling_QNetwork(state_space,action_space)
        self.qnetwork_target = Dueling_QNetwork(state_space,action_space)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(),lr=learning_rate)
        # Initialize replaybuffer
        self.memory = PriorityReplayBuffer(action_size,buffer_size,batch_size,seed,alpha)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        
    def step(self,state,action,reward,next_state,done):
        # Calculate TD error

        # Target
        current_network_action = self.local_network(next_state).max(1)[1]
        target = reward + self.GAMMA*(self.qnetwork_target(next_state)[current_network_action])
        # Local
        local = self.local_network(state)[action]
        TD_error = reward + target - local
        # current_network_action = np.argmax(self.local_network(next_state))
        # TD_error = reward + self.GAMMA*(self.target_network(next_state)[current_network_action]) - self.local_network(state)[action]
        # priority = (abs(TD_error) + self.epsilon)**self.alpha
        # # self.learning_rate * importance

        # TD_error = reward + self.GAMMA*(self.target_network(next_state)[current_network_action]) - self.local_network(state)[action]
        # priority = (abs(TD_error) + self.epsilon)**self.alpha
        # probability = priority / self.memory.TD_sum
        # Importance weights
        # importance = (priority * len(self.memory))**-self.beta
        # self.max_w = max(self.max_w,importance)
        # norm_importance = importance / self.max_w
        # Save the experience
        self.memory.add(state,action,reward,next_state,done,TD_error)
        
        # learn from the experience
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            if len(self.memory) > self.buffer_size:
                experiences,indicies,weights = self.memory.sample()
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
            return random.sample(np.arange(self.action_space))
        
    def learn(self,experiences,indicies,weights):
        
        states,actions,rewards,next_states,dones = experiences
        # Local max action
        local_next_state_actions = self.local_network(next_states).max(1)[1].unsqueeze(1)
        # Target
        target_values = self.qnetwork_target(next_states).detach()
        max_target = target_values[np.arange(self.batch_size),local_next_state_actions]
        max_target *= (1-dones) 
        targets = rewards + (self.GAMMA*max_target)
        # targets = rewards + self.GAMMA*(target_values.gather(1,local_next_state_actions))
        # Local
        local = self.local_network(states).gather(1,actions)
        TD_error = local - targets
        loss = (weights * TD_error)**2*0.5.mean()
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.local_network.parameters(),self.clip_norm)
        self.optimizer.step()
        # Update the priorities
        TD_errors = np.abs(TD_error.detach().cpu().numpy())
        self.memory.sum_tree.update_priorities(priorities,indicies)
        self.update_target()

    # def update_target(self,tau):
    #     for local_param,target_param in zip(self.qnetwork_local.parameters(),self.qnetwork_target.parameters()):
    #         target_param.data.copy_(local_param.data)
        
    # Polyak averaging  
    def update_target(self,tau):
        for local_param,target_param in zip(self.qnetwork_local.parameters(),self.qnetwork_target.parameters()):
            local_param.data.copy_(tau*local_param.data + (1-tau)*target_param.data)
#         self.qnetwork_local.parameters() = TAU*self.qnetwork_local.parameters() + (1-TAU)*self.qnetwork_target.parameters()