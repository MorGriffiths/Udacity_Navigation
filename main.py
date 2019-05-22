from train import train
from Agents import DQN,Priority_DQN
import matplotlib.pyplot as plt
import gym
import numpy as np

# Parameters
BUFFER_SIZE = 10000
BATCH_SIZE = 32
ALPHA = 0.5
START_BETA = 0.5
END_BETA = 1
LR = 0.000025
EPSILON = 1
MIN_EPSILON = 0.1
GAMMA = 0.99
TAU = 1e-3
UPDATE_EVERY = 4

def main():
    env = gym.make('LunarLander-v2')
    env.seed(0)
    nA = env.action_space.n
    nS = env.observation_space.shape[0]
    seed = 7
    agent = Priority_DQN(nA,nS,seed)

    scores = train(agent,env)

    # plot the scores
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.savefig('agent_scores.png', bbox_inches='tight')

if __name__ == "__main__":
    main()