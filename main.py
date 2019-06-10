
import matplotlib.pyplot as plt
import numpy as np
import sys

from train import train
from Agents.Priority_DQN import Priority_DQN
from plot import plot

from unityagents import UnityEnvironment
# from gym_unity.envs import UnityEnv

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
    env_name = file_name="Environments/Banana.app"
    train_mode = True  # Whether to run the environment in training or inference mode
    env = UnityEnvironment(file_name=env_name)
    # env = UnityEnvironment(file_name="/data/Banana_Linux_NoVis/Banana.x86_64")
    # Set the default brain to work with
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    env_info = env.reset(train_mode=True)[brain_name]
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