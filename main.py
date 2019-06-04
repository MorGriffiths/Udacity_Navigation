
import matplotlib.pyplot as plt
import gym
import numpy as np
import sys

from train import train
from Agents.Priority_DQN import Priority_DQN
from plot import plot

sys.path.append('/Users/morgan/Code/ml-agents/ml-agents-envs/mlagents/')
from envs.environment import UnityEnvironment

# Parameters
BUFFER_SIZE = 10000
MIN_BUFFER_SIZE = 200
BATCH_SIZE = 50
ALPHA = 0.6 # 0.7 or 0.6
START_BETA = 0.5 # from 0.5-1
END_BETA = 1
LR = 0.00025
EPSILON = 1
MIN_EPSILON = 0.01
GAMMA = 0.99
TAU = 0.01
UPDATE_EVERY = 4
CLIP_NORM = 10

def main():
    env_name = "BananaCollectors"
    env = UnityEnvironment(file_name="/data/Banana_Linux_NoVis/Banana.x86_64")
    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    # Action and Observation spaces
    nA = brain.vector_action_space_size
    nS = env_info.vector_observations.shape[1]
    print('Observation Space {}, Action Space {}'.format(nS,nA))
    seed = 7
    agent = Priority_DQN(nS,nA,seed,UPDATE_EVERY,BATCH_SIZE,BUFFER_SIZE,MIN_BUFFER_SIZE,LR,GAMMA,TAU,CLIP_NORM,ALPHA)

    scores = train(agent,env,brain_name)

    # plot the scores
    plot(scores)
    
if __name__ == "__main__":
    main()