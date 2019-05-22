from train import train
from agent import Agent
import matplotlib.pyplot as plt

def main():
    env = gym.make('LunarLander-v2')
    env.seed(0)
    nA = env.action_space.n
    nS = env.observation_space.shape[0]
    seed = 7
    agent = Agent(nA,nS,seed)

    scores = train()

    # plot the scores
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    savefig('/Users/morgan/Code/Udacity_Navigation/agent_scores.png', bbox_inches='tight')

if __name__ == if __name__ == "__main__":
    main()