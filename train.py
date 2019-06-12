import torch
import numpy as np
from collections import deque

def train(agent,env,brain_name,checkpoint_path,n_episodes=20, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    """Deep Q-Learning.
    
    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
        Instead of updating target every (int) steps, using Polyak updating of .1 to gradually merge the networks
    """
    scores = []
    scores_window = deque(maxlen=100)
    eps = eps_start
    index = 0
    for i_episode in range(1,n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.visual_observations[0].transpose([-1,0,1,2])
        score = 0
        for t in range(max_t):
            action = agent.act(state,eps)
            env_info = env.step(action)[brain_name]        # send the action to the environment
            # Necessary to reshape the vector for torch conv layers
            next_state = env_info.visual_observations[0].transpose([-1,0,1,2])
            reward = env_info.rewards[0]
            done = env_info.local_done[0]
            agent.step(state,action,reward,next_state,done,index)
            state = next_state
            score += reward
            index += 1
            if done:
                break
        scores_window.append(score)
        scores.append(score)
        eps = max(eps*eps_decay,eps_end)
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)),end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window) >= 13.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), checkpoint_path)
            break
    return scores